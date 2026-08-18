"""
p20_mi_probe_training.py

Stage 1 of "Option C": trains a window-level probe (same LinearProbeHead/MLPProbeHead classes as
p08b) under the standard multiple-instance-learning (MI) assumption, rather than the "collective"
assumption p08b's naive per-window training uses today.

--------------------------------------------------------------------------
Why: the flaw in the current probe's training that motivated this
--------------------------------------------------------------------------
p08b trains the probe by copying each subject's label onto every one of their windows and applying
plain per-window cross-entropy. This is the "collective" MI assumption -- every instance in a bag
shares the bag's label -- and it is known to be the wrong assumption for exactly this kind of data: a
schizophrenia patient can have plenty of ordinary-looking sleep windows, so forcing the classifier to
call ALL of a patient's windows positive injects real label noise into training. The "standard" MI
assumption is more realistic: a NEGATIVE bag's instances are safely assumed to be uniformly negative
(a true control has no pathological windows), but a POSITIVE bag only guarantees that AT LEAST ONE
instance is positive -- the rest can look arbitrarily normal.

--------------------------------------------------------------------------
How this avoids Option B's obscurity problem, structurally, not by hope
--------------------------------------------------------------------------
Option A's causal signal stayed traceable specifically because the probe was trained to convergence
BEFORE any learned aggregation (attention) ever saw it -- there was no second learnable module for it
to jointly co-adapt with. This script preserves that property by using a FIXED, parameter-free
aggregation (max, or mean-of-top-k) for positive bags during training, not a learned one:

    negative bag:  loss = mean_i  BCE(window_prob_i, 0)             -- every window supervised directly
    positive bag:  loss = BCE(topk_mean_i(window_prob_i), 1)        -- only the model's own current
                                                                        best-scoring window(s) get
                                                                        pushed toward positive; max/
                                                                        top-k has no learnable
                                                                        parameters of its own to
                                                                        collude with the scorer

Because max/top-k is not learnable, there is nothing here for the scorer to jointly compensate with --
unlike Option B, where the scorer and the attention weights co-adapted freely. Stage 2 (training
attention on top of THIS probe, once frozen) is unchanged from Option A's existing
p13_attention_mil_pooling.py -- just point --probe-checkpoint at whatever this script saves, since the
checkpoint format matches p08b's exactly.

--------------------------------------------------------------------------
Practical asymmetry this training loop has to account for
--------------------------------------------------------------------------
A negative bag contributes a loss term averaged over potentially hundreds of windows; a positive bag
contributes one term derived from only top_k windows. That's a real imbalance in how much gradient
signal each class provides per subject, on top of whatever class-balance imbalance already exists --
--pos-loss-weight exists to rebalance it; there is no principled default here, it needs tuning.

Usage:
    python p20_mi_probe_training.py \
        --cache-dir /data/eeg_study/cache \
        --train-manifest /data/eeg_study/train_manifest.csv \
        --val-manifest /data/eeg_study/val_manifest.csv \
        --head-type linear --top-k 1 --pos-loss-weight 1.0 --epochs 40
"""

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cbramod_utils import setup_logger
from cbramod_common import (
    CachedFeatureSubjectDataset,
    CBraModTrainer,
    LinearProbeHead,
    MLPProbeHead,
    add_log_filename_argument,
    flatten_cached_feature_dataset,
    is_checkpoint_improvement,
    load_subject_ids,
    seed_everything,
    setup_cache_cli_parser,
    setup_training_cli_parser,
)


def parse_cli_args() -> argparse.Namespace:
    parser = setup_training_cli_parser(
        description="Stage 1 of Option C: standard-MI probe training (asymmetric loss, no learned "
                    "aggregation) -- produces a p08b-compatible checkpoint for p13/p14/p09f/p09h/p09i."
    )

    setup_cache_cli_parser(parser)

    mi_group = parser.add_argument_group("Standard-MI Training")
    mi_group.add_argument(
        "--top-k", type=int, default=1,
        help="Number of highest-scoring windows averaged for a positive bag's aggregate loss term "
             "(1 = pure max, the most literal reading of the standard-MI assumption; >1 trades some "
             "theoretical purity for a smoother gradient signal)."
    )
    mi_group.add_argument(
        "--pos-loss-weight", type=float, default=1.0,
        help="Multiplier on the positive-bag loss term, to compensate for it being derived from only "
             "top_k windows vs. a negative bag's loss averaged over its whole (much larger) window set. "
             "No principled default -- tune against validation."
    )
    mi_group.add_argument(
        "--subjects-per-step", type=int, default=8,
        help="Number of subjects (bags) whose losses get averaged before each optimizer.step() -- "
             "gradient accumulation, same rationale as p13/p16's flag of the same name."
    )

    add_log_filename_argument(parser, __file__)

    args = parser.parse_args()
    return args


def mi_loss_for_subject(window_probs_pos: torch.Tensor, label: int, top_k: int, pos_loss_weight: float) -> torch.Tensor:
    """
    Asymmetric standard-MI loss for one subject's bag of window-level P(class=1) values.

    label == 0 (negative bag): every window supervised directly with target 0 -- safe under the
    standard-MI assumption (a true control has no pathological windows).
    label == 1 (positive bag): only the mean of the top_k highest-scoring windows is supervised
    with target 1 -- a FIXED, parameter-free aggregation (no learnable weights), so there is nothing
    for the scorer to jointly co-adapt with the way Option B's learned attention could.
    """
    eps = 1e-6
    if label == 0:
        # F.binary_cross_entropy's default reduction ("mean") already averages over every window in
        # the bag -- exactly the "every window supervised directly" behavior this branch wants.
        return F.binary_cross_entropy(window_probs_pos.clamp(eps, 1 - eps), torch.zeros_like(window_probs_pos))

    k = min(top_k, window_probs_pos.shape[0])
    topk_vals, _ = torch.topk(window_probs_pos, k)
    aggregate = topk_vals.mean()
    target = torch.ones((), device=window_probs_pos.device)
    return pos_loss_weight * F.binary_cross_entropy(aggregate.clamp(eps, 1 - eps), target)


class MITrainer(CBraModTrainer):
    """Trains a window-level probe head under the standard-MI assumption; reuses
    CBraModTrainer.evaluate_subject_pooling() for validation (identical multi-strategy subject-level
    pooling/threshold-tuning to p08b, for direct comparability and checkpoint-format compatibility)."""

    def __init__(self, config: argparse.Namespace, logger):
        super().__init__(config, logger)

    def train(self, master_cache_path: Path) -> dict:
        train_subject_ids = load_subject_ids(Path(self.config.train_manifest))
        val_subject_ids = load_subject_ids(Path(self.config.val_manifest))

        self.logger.info(f"Loading cached feature tensors into RAM from {master_cache_path}...")
        train_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=train_subject_ids)
        val_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=val_subject_ids)

        leaked = set(train_ds.unique_subjects) & set(val_ds.unique_subjects)
        assert not leaked, f"[CRITICAL LEAKAGE] {len(leaked)} subject(s) in both train and val: {leaked}"
        self.logger.info(f"✓ [Leak Check Passed] Train: {len(train_ds)} subjects | Val: {len(val_ds)} subjects.")

        if self.config.head_type == "linear":
            head = LinearProbeHead(num_patches=self.config.num_patches, emb_dim=self.config.cbra_dim, num_classes=self.config.num_classes)
        elif self.config.head_type == "mlp":
            head = MLPProbeHead(
                num_patches=self.config.num_patches, emb_dim=self.config.cbra_dim,
                hidden_dim=self.config.head_dim, num_classes=self.config.num_classes, dropout=self.config.dropout,
            )
        else:
            raise ValueError(f"Unknown --head-type: {self.config.head_type}")
        head.to(self.device)

        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config.head_lr, weight_decay=self.config.weight_decay)

        best_primary_f1, best_primary_auc = 0.0, 0.0
        best_thresholds, best_primary_metrics = {}, {}
        patience_counter = 0
        best_model_path = Path(self.config.checkpoint_dir) / self.config.checkpoint_filename if self.config.checkpoint_dir else Path(self.config.checkpoint_filename)

        self.logger.info(
            f"Starting Standard-MI Probe Training ({self.config.epochs} epochs max | top_k={self.config.top_k} | "
            f"pos_loss_weight={self.config.pos_loss_weight} | subjects/step: {self.config.subjects_per_step})"
        )
        self.logger.info("=" * 125)

        rng = np.random.default_rng(self.config.seed)
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            head.train()
            subject_order = rng.permutation(len(train_ds))
            total_loss, n_subjects = 0.0, 0
            optimizer.zero_grad()
            for i, subj_idx in enumerate(subject_order):
                bag_feats, label, _subject_id, _stages, _indices = train_ds[subj_idx]
                bag_feats = bag_feats.to(self.device).float()

                window_logits = head(bag_feats)
                window_probs_pos = torch.softmax(window_logits, dim=1)[:, 1]
                loss = mi_loss_for_subject(window_probs_pos, int(label.item()), self.config.top_k, self.config.pos_loss_weight)

                (loss / self.config.subjects_per_step).backward()
                n_subjects += 1
                if (i + 1) % self.config.subjects_per_step == 0 or (i + 1) == len(subject_order):
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item()
            train_loss = total_loss / max(n_subjects, 1)

            # Validation: run the head over flattened val windows IN BATCHES (same batch_size-driven
            # DataLoader pattern p08b uses, not one unbatched forward pass over the whole val set --
            # embeddings are small per-window, but a full val set can still be tens of thousands of
            # windows, and there's no reason to risk GPU memory on that when batching is free) and
            # reuse the SAME multi-strategy subject-level pooling evaluation p08b uses, for direct
            # comparability and a checkpoint format p13/p14/p09f/p09h/p09i can all read unmodified.
            head.eval()
            val_feats, val_labels, val_subject_ids_flat, _, _ = flatten_cached_feature_dataset(val_ds)
            val_loader = DataLoader(
                TensorDataset(val_feats, val_labels), batch_size=self.config.batch_size,
                shuffle=False, num_workers=self.config.num_workers, pin_memory=True,
            )
            val_probs_list = []
            with torch.no_grad():
                for x_b, _y_b in val_loader:
                    x_b = x_b.to(self.device, non_blocking=True).float()
                    val_probs_list.append(torch.softmax(head(x_b), dim=1).cpu().numpy())
            val_probs = np.concatenate(val_probs_list, axis=0)
            pooling_results = self.evaluate_subject_pooling(
                val_probs=val_probs, val_targets=val_labels.numpy(), val_subject_ids=val_subject_ids_flat
            )

            primary_metrics = pooling_results[self.config.primary_pooling]
            primary_f1 = primary_metrics["subject_macro_f1"]
            primary_auc = primary_metrics["roc_auc"]
            elapsed = time.time() - t0

            log_str = (
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.2f}s) | Train Loss: {train_loss:.4f} | "
                f"Subj F1 ({self.config.primary_pooling}): {primary_f1:.4f} | Subj AUC: {primary_auc:.4f}"
            )

            if is_checkpoint_improvement(primary_f1, primary_auc, best_primary_f1, best_primary_auc):
                best_primary_f1, best_primary_auc = primary_f1, primary_auc
                best_primary_metrics = primary_metrics
                best_thresholds = {strat: res["optimal_threshold"] for strat, res in pooling_results.items()}
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": head.state_dict(),
                        "checkpoint_kind": "head_only",
                        "head_type": self.config.head_type,
                        "head_dim": self.config.head_dim,
                        "num_patches": self.config.num_patches,
                        "cbra_dim": self.config.cbra_dim,
                        "num_classes": self.config.num_classes,
                        "num_channels": self.config.num_channels,
                        "sfreq": self.config.sfreq,
                        "best_macro_f1": best_primary_f1,
                        "best_auc": best_primary_auc,
                        "primary_pooling": self.config.primary_pooling,
                        "top_percentile": self.config.top_percentile,
                        "t_window": self.config.t_window,
                        "optimal_thresholds": best_thresholds,
                        "pooling_summary": pooling_results,
                        "mi_top_k": self.config.top_k,
                        "mi_pos_loss_weight": self.config.pos_loss_weight,
                    },
                    best_model_path,
                )
                log_str += " --> [BEST MODEL SAVED]"
            else:
                patience_counter += 1
                log_str += f" | EarlyStop: {patience_counter}/{self.config.patience}"

            self.logger.info(log_str)
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.logger.info("=" * 125)
        self.logger.info(
            f"Training Complete. Best Validation Subject Macro F1 ({self.config.primary_pooling}): "
            f"{best_primary_f1:.4f} | Best AUC: {best_primary_auc:.4f}"
        )
        self.logger.info(f"Calibrated Strategy Thresholds: {best_thresholds}")
        return best_primary_metrics


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)

    if not args.train_manifest or not args.val_manifest:
        raise ValueError("--train-manifest and --val-manifest are both required.")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    trainer = MITrainer(args, logger)
    trainer.train(master_cache_path)


if __name__ == "__main__":
    main()
