"""
p13_attention_mil_pooling.py

Option A of the attention-based MIL follow-up: replace the fixed p85-percentile subject-level
pooling rule with a LEARNED attention-weighted aggregation, while changing nothing else -- the
CBraMod backbone stays frozen, and the window-level probe head (trained separately by
p08b_finetune_probing.py) stays frozen too. Only the aggregation step becomes learned.

--------------------------------------------------------------------------
No fixed number of windows anywhere in this architecture
--------------------------------------------------------------------------
A subject's recording is a "bag" of however many windows it has -- this varies subject to subject,
and nothing here assumes a fixed bag size:

  - AttentionPoolingHead.gate is an MLP applied independently, per-window, with weights SHARED
    across all windows in the bag (same nn.Linear layers regardless of bag size).
  - torch.softmax(scores, dim=0) normalizes over whatever length that call's `scores` tensor
    happens to be -- 200 windows for a short recording, 900 for a long one. There is no fixed-size
    input anywhere in the module.
  - Training loops over subjects ONE AT A TIME (batch size 1 at the bag level), so no padding or
    masking is needed to handle subjects with different window counts in the same "batch" --
    gradients are simply accumulated over --subjects-per-step subjects before each optimizer.step().

This is what makes attention-MIL (Ilse et al., 2018) a natural fit for variable-length recordings,
in contrast to an architecture that assumes a fixed input size (e.g. a fixed-length feature vector
or a sequence model built for one specific sequence length).

Usage:
    python p13_attention_mil_pooling.py \
        --cache-dir /data/eeg_study/cache \
        --train-manifest /data/eeg_study/train_manifest.csv \
        --val-manifest /data/eeg_study/val_manifest.csv \
        --probe-checkpoint /data/eeg_study/checkpoints-probe-linear/cbramod_ckpt.pt \
        --epochs 40 --attn-lr 1e-3
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from cbramod_utils import setup_logger
from cbramod_common import (
    CachedFeatureSubjectDataset,
    LinearProbeHead,
    MLPProbeHead,
    compute_pooled_scores,
    is_checkpoint_improvement,
    seed_everything,
    setup_training_cli_parser,
)


# =====================================================================
# 1. CLI
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = setup_training_cli_parser(
        description="Attention-MIL subject-level pooling (Option A: learned aggregation over a "
                     "frozen window-level probe's outputs) -- reads a master cache built by "
                     "p08a_extract_features.py and a probe checkpoint trained by p08b_finetune_probing.py"
    )

    cache_group = parser.add_argument_group("Cache Controls")
    cache_group.add_argument("--cache-dir", type=str, required=True, help="Directory containing the master cache")
    cache_group.add_argument(
        "--master-cache-name", type=str, default="cached_master_embeddings.pt",
        help="Filename of the whole-cohort cached embeddings file (see p08a_extract_features.py)"
    )

    probe_group = parser.add_argument_group("Frozen Probe (window-level)")
    probe_group.add_argument(
        "--probe-checkpoint", type=str, required=True,
        help="Path to a probe head checkpoint saved by p08b_finetune_probing.py. Its architecture "
             "(--head-type/--head-dim/--num-patches/--cbra-dim) must match what THIS run's CLI args "
             "specify, since the checkpoint only stores weights, not the architecture."
    )

    attn_group = parser.add_argument_group("Attention Pooling Head")
    attn_group.add_argument("--attn-hidden-dim", type=int, default=64, help="Hidden dim of the attention gate MLP")
    attn_group.add_argument("--attn-dropout", type=float, default=0.1, help="Dropout inside the attention gate")
    attn_group.add_argument("--attn-lr", type=float, default=1e-3, help="Learning rate for the attention gate (AdamW)")
    attn_group.add_argument(
        "--subjects-per-step", type=int, default=8,
        help="Number of subjects (bags) whose losses get averaged before each optimizer.step() -- "
             "a form of gradient accumulation that smooths single-subject noise WITHOUT requiring "
             "padding/masking across bags of different sizes."
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-filename", type=str, default=Path(__file__).stem + ".log", help="Filename for pipeline log output")

    args = parser.parse_args()
    return args


# =====================================================================
# 2. ATTENTION POOLING HEAD
# =====================================================================

class AttentionPoolingHead(nn.Module):
    """
    Learned replacement for compute_pooled_scores(method="p85_score"). Given one subject's bag of
    window embeddings and that same subject's FROZEN window-level probabilities (already computed
    by the probe head this script does not retrain), learns a per-window attention weight and
    returns the attention-weighted sum of the window probabilities as the subject-level score.

    The gate conditions on the embedding (so it can learn what kind of window to trust -- e.g. the
    causal investigation's sigma-band finding suggests the model already has an implicit notion of
    "this window's spectral content is informative"), but the quantity being pooled is still the
    already-validated, already-trained window-level probability -- this keeps the pooled score
    directly comparable to every prior pooling strategy (p85_score, top_10_mean, ...).
    """
    def __init__(self, num_patches: int = 30, emb_dim: int = 200, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        in_features = num_patches * emb_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bag_feats: torch.Tensor, window_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bag_feats:    [n_windows, num_patches, emb_dim] -- ONE subject's windows. n_windows varies
                      call to call; nothing about self.gate's parameters depends on it.
        window_probs: [n_windows] -- frozen probe's P(class=1) per window, already computed
                      upstream with no_grad().

        Returns (subject_prob, attn_weights): subject_prob is a 0-dim tensor, attn_weights is
        [n_windows] (sums to 1) -- exposed for inspection/plotting, same role as the leave-one-out
        contribution and LOO-influence analyses used to characterize p85 pooling.
        """
        n_windows = bag_feats.shape[0]
        flat = bag_feats.reshape(n_windows, -1)          # [n_windows, num_patches * emb_dim]
        scores = self.gate(flat).squeeze(-1)              # [n_windows]
        attn_weights = torch.softmax(scores, dim=0)        # normalizes over THIS bag's actual size
        subject_prob = (attn_weights * window_probs).sum()
        return subject_prob, attn_weights


# =====================================================================
# 3. WINDOW-LEVEL PROBE (frozen) FORWARD PASS
# =====================================================================

def build_frozen_probe(config: argparse.Namespace, device: torch.device) -> nn.Module:
    """Reconstructs the probe head architecture and loads frozen weights from --probe-checkpoint."""
    if config.head_type == "linear":
        probe = LinearProbeHead(num_patches=config.num_patches, emb_dim=config.cbra_dim, num_classes=config.num_classes)
    elif config.head_type == "mlp":
        probe = MLPProbeHead(
            num_patches=config.num_patches, emb_dim=config.cbra_dim,
            hidden_dim=config.head_dim, num_classes=config.num_classes, dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown --head-type: {config.head_type}")

    ckpt = torch.load(config.probe_checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


@torch.no_grad()
def frozen_window_probs(probe: nn.Module, bag_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    """P(class=1) per window, from the frozen probe, for one subject's bag of windows."""
    logits = probe(bag_feats.to(device).float())
    return torch.softmax(logits, dim=1)[:, 1]


# =====================================================================
# 4. THRESHOLD TUNING (mirrors CBraModTrainer.evaluate_subject_pooling's per-strategy sweep)
# =====================================================================

def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Sweeps 99 thresholds for the macro-F1-optimal subject-level decision boundary."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    final_preds = (scores >= best_t).astype(int)
    return {
        "subject_macro_f1": best_f1,
        "optimal_threshold": float(best_t),
        "subject_accuracy": accuracy_score(labels, final_preds),
        "subject_sensitivity": recall_score(labels, final_preds),
        "subject_specificity": recall_score(labels, final_preds, pos_label=0),
        "roc_auc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5,
    }


# =====================================================================
# 5. TRAIN / EVAL LOOPS
# =====================================================================

def run_epoch_train(
    attn_head: AttentionPoolingHead, probe: nn.Module, dataset: CachedFeatureSubjectDataset,
    optimizer: torch.optim.Optimizer, device: torch.device, subjects_per_step: int, subject_order: np.ndarray,
) -> float:
    """
    One training epoch, iterating bags one subject at a time (see module docstring for why this
    sidesteps the variable-bag-size batching problem entirely). Losses are averaged over groups of
    `subjects_per_step` subjects before each optimizer.step() -- gradient accumulation, not batching
    of the underlying tensors, so no bag ever needs to be padded to match another bag's length.
    """
    attn_head.train()
    total_loss, n_subjects = 0.0, 0
    optimizer.zero_grad()
    for i, subj_idx in enumerate(subject_order):
        bag_feats, label, _subject_id, _stages, _indices = dataset[subj_idx]
        bag_feats = bag_feats.to(device).float()
        label = label.to(device).float()

        window_probs = frozen_window_probs(probe, bag_feats, device)
        subject_prob, _attn_weights = attn_head(bag_feats, window_probs)
        loss = F.binary_cross_entropy(subject_prob.clamp(1e-6, 1 - 1e-6), label)

        (loss / subjects_per_step).backward()
        n_subjects += 1

        if (i + 1) % subjects_per_step == 0 or (i + 1) == len(subject_order):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / max(n_subjects, 1)


@torch.no_grad()
def run_epoch_eval(
    attn_head: AttentionPoolingHead, probe: nn.Module, dataset: CachedFeatureSubjectDataset, device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (attention_pooling_metrics, baseline_p85_metrics) for direct, apples-to-apples
    comparison -- both computed from the SAME frozen window-level probabilities, differing only in
    how they're aggregated into one subject-level score.
    """
    attn_head.eval()
    attn_scores, p85_scores, labels = [], [], []
    for subj_idx in range(len(dataset)):
        bag_feats, label, _subject_id, _stages, _indices = dataset[subj_idx]
        bag_feats = bag_feats.to(device).float()

        window_probs = frozen_window_probs(probe, bag_feats, device)
        subject_prob, _attn_weights = attn_head(bag_feats, window_probs)

        attn_scores.append(subject_prob.item())
        p85_scores.append(compute_pooled_scores(window_probs.cpu().numpy(), method="p85_score"))
        labels.append(int(label.item()))

    labels = np.array(labels)
    attn_metrics = tune_threshold(np.array(attn_scores), labels)
    p85_metrics = tune_threshold(np.array(p85_scores), labels)
    return attn_metrics, p85_metrics


# =====================================================================
# 6. MAIN
# =====================================================================

def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.train_manifest or not args.val_manifest:
        raise ValueError("--train-manifest and --val-manifest are both required (fixed split only in this first version).")

    def load_subject_ids(manifest_csv: str) -> List[str]:
        df = pd.read_csv(manifest_csv)
        return df["subject_id"].astype(str).tolist()

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    train_subject_ids = load_subject_ids(args.train_manifest)
    val_subject_ids = load_subject_ids(args.val_manifest)

    logger.info(f"Loading cached feature tensors into RAM from {master_cache_path}...")
    train_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=train_subject_ids)
    val_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=val_subject_ids)

    leaked = set(train_ds.unique_subjects) & set(val_ds.unique_subjects)
    assert not leaked, f"[CRITICAL LEAKAGE] {len(leaked)} subject(s) in both train and val: {leaked}"

    logger.info(
        f"✓ [Leak Check Passed] Train: {len(train_ds)} subjects | Val: {len(val_ds)} subjects. "
        f"Bag sizes are NOT fixed -- window counts per subject vary freely."
    )

    probe = build_frozen_probe(args, device)
    logger.info(f"Loaded frozen probe from {args.probe_checkpoint} (head_type={args.head_type}); probe parameters are NOT trained here.")

    attn_head = AttentionPoolingHead(
        num_patches=args.num_patches, emb_dim=args.cbra_dim,
        hidden_dim=args.attn_hidden_dim, dropout=args.attn_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(attn_head.parameters(), lr=args.attn_lr, weight_decay=args.weight_decay)

    best_f1, best_auc = 0.0, 0.0
    patience_counter = 0
    best_model_path = Path(args.checkpoint_dir) / args.checkpoint_filename if args.checkpoint_dir else Path(args.checkpoint_filename)

    logger.info(f"Starting Attention-MIL Pooling Training ({args.epochs} epochs max | subjects/step: {args.subjects_per_step})")
    logger.info("=" * 125)

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        subject_order = rng.permutation(len(train_ds))
        train_loss = run_epoch_train(attn_head, probe, train_ds, optimizer, device, args.subjects_per_step, subject_order)
        attn_metrics, p85_metrics = run_epoch_eval(attn_head, probe, val_ds, device)
        elapsed = time.time() - t0

        log_str = (
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.2f}s) | Train Loss: {train_loss:.4f} | "
            f"Attn: F1={attn_metrics['subject_macro_f1']:.4f} AUC={attn_metrics['roc_auc']:.4f} | "
            f"p85 (same frozen probs): F1={p85_metrics['subject_macro_f1']:.4f} AUC={p85_metrics['roc_auc']:.4f}"
        )

        if is_checkpoint_improvement(attn_metrics["subject_macro_f1"], attn_metrics["roc_auc"], best_f1, best_auc):
            best_f1, best_auc = attn_metrics["subject_macro_f1"], attn_metrics["roc_auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "attn_head_state_dict": attn_head.state_dict(),
                    "best_macro_f1": best_f1,
                    "best_auc": best_auc,
                    "attn_metrics": attn_metrics,
                    "p85_metrics_same_epoch": p85_metrics,
                },
                best_model_path,
            )
            log_str += " --> [BEST ATTENTION MODEL SAVED]"
        else:
            patience_counter += 1
            log_str += f" | EarlyStop: {patience_counter}/{args.patience}"

        logger.info(log_str)
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info("=" * 125)
    logger.info(f"Training Complete. Best Attention-Pooling Subject Macro F1: {best_f1:.4f} | Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
