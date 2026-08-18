"""
p13_attention_mil_pooling.py

A hybrid, closer to Option A than Option B, of the attention-based MIL follow-up: replace the fixed
p85-percentile subject-level pooling rule with a LEARNED attention-weighted aggregation, while the
CBraMod backbone and the window-level probe head (trained separately by p08b_finetune_probing.py)
both stay frozen and untouched.

To be precise about where this sits relative to the original A/B framing: the quantity being
POOLED is still the probe head's own scalar window-level probability, exactly as pure Option A
described -- the probe is never retrained, and the pooled score stays directly comparable to every
other pooling strategy in evaluate_subject_pooling(). But the attention GATE that decides each
window's weight conditions on the full frozen embedding, not on that scalar alone (see
AttentionPoolingHead's docstring for why: a 1-dimensional input to the gate would only let it learn
a monotonic-ish reweighting of the probe's own score, degenerating into "yet another fixed pooling
statistic" rather than genuinely contextual attention). That's a deliberate, but real, departure
from "pure" Option A -- it pulls the gate's own capacity/overfitting-risk profile partway toward
Option B's end of the spectrum, even though the pooled target and the frozen probe are unchanged.

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
    # Train (fixed split), then score the BEST saved checkpoint against held-out test:
    python p13_attention_mil_pooling.py \
        --cache-dir /data/eeg_study/cache \
        --train-manifest /data/eeg_study/train_manifest.csv \
        --val-manifest /data/eeg_study/val_manifest.csv \
        --test-manifest /data/eeg_study/test_manifest.csv \
        --probe-checkpoint /data/eeg_study/checkpoints-probe-linear/cbramod_ckpt.pt \
        --epochs 40 --attn-lr 1e-3

    # Re-evaluate a previously-saved attention-head checkpoint, no training:
    python p13_attention_mil_pooling.py \
        --cache-dir /data/eeg_study/cache \
        --probe-checkpoint /data/eeg_study/checkpoints-probe-linear/cbramod_ckpt.pt \
        --resume-checkpoint /data/eeg_study/checkpoints-attn/best_attn.pt --eval-only \
        --test-manifest /data/eeg_study/test_manifest.csv
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from cbramod_utils import setup_logger
from cbramod_common import (
    CachedFeatureSubjectDataset,
    add_log_filename_argument,
    build_frozen_probe,
    compute_pooled_scores,
    is_checkpoint_improvement,
    seed_everything,
    setup_cache_cli_parser,
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

    setup_cache_cli_parser(parser)

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

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--test-manifest", type=str, default=None,
        help="Optional p03 test_manifest.csv. If given, after training the BEST saved checkpoint "
             "(not whatever the last epoch happened to leave in memory -- early stopping means those "
             "can differ) is reloaded and scored against these held-out subjects, attention vs. p85, "
             "the same side-by-side comparison as every validation epoch."
    )
    eval_group.add_argument(
        "--eval-only", action="store_true",
        help="Skip training entirely; just load --resume-checkpoint and evaluate against "
             "--val-manifest (and --test-manifest, if given)."
    )
    eval_group.add_argument(
        "--resume-checkpoint", type=str, default=None,
        help="Path to a previously-saved attention-head checkpoint (this script's own output). "
             "Required with --eval-only; also used to reload the actual best checkpoint for the "
             "post-training --test-manifest pass."
    )

    add_log_filename_argument(parser, __file__)

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

    The gate conditions on the full embedding rather than on the probe's scalar output alone: a
    window-level probability is already a heavy compression of a ~num_patches*emb_dim-dimensional
    embedding down to one number, optimized for a different objective (window-level classification).
    Two windows can land at the same probe output (e.g. both near 0.5) for very different reasons --
    one genuinely ambiguous-but-clean, one noisy/borderline-artifactual -- and that distinction is
    already erased by the time a scalar-only gate would see it. Conditioning on the embedding
    instead gives the gate a chance to learn "how much to trust this window" using information the
    probe's own compression discarded; a gate restricted to a 1-dimensional input would only be able
    to learn some monotonic-ish reweighting of the probe's own score, which is uncomfortably close
    to just being another fixed pooling statistic rather than genuinely contextual attention.

    The quantity being POOLED, though, is still the already-validated, already-trained window-level
    probability, not the embedding itself -- this keeps the pooled score directly comparable to
    every prior pooling strategy (p85_score, top_10_mean, ...) and keeps the probe head itself
    completely untouched. See this file's module docstring for where that puts this design relative
    to the original Option A / Option B framing (a deliberate hybrid, not pure Option A).
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


@torch.no_grad()
def frozen_window_probs(probe: nn.Module, bag_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    """P(class=1) per window, from the frozen probe, for one subject's bag of windows."""
    logits = probe(bag_feats.to(device).float())
    return torch.softmax(logits, dim=1)[:, 1]


# =====================================================================
# 4. THRESHOLD TUNING (mirrors CBraModTrainer.evaluate_subject_pooling's per-strategy sweep)
# =====================================================================

def tune_threshold(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Sweeps 99 thresholds for the macro-F1-optimal subject-level decision boundary -- UNLESS a fixed
    `threshold` is passed in (one already selected on validation), in which case it's applied as-is
    with no sweep. This distinction matters: sweeping against a split's own labels and then reporting
    F1/accuracy/sensitivity/specificity computed at that just-found threshold is legitimate model
    SELECTION on validation, but doing the same thing on held-out test silently tunes the decision
    boundary against the very labels being used to report performance -- inflating every
    threshold-dependent metric. AUC is unaffected either way (it's rank-based, no threshold involved),
    which is exactly why is_checkpoint_improvement() treats AUC as the more trustworthy of the two.
    """
    if threshold is None:
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    else:
        best_t = threshold

    final_preds = (scores >= best_t).astype(int)
    return {
        "subject_macro_f1": f1_score(labels, final_preds, average="macro", zero_division=0),
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
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (attention_pooling_metrics, baseline_p85_metrics) for direct, apples-to-apples
    comparison -- both computed from the SAME frozen window-level probabilities, differing only in
    how they're aggregated into one subject-level score.

    fixed_thresholds: (attn_threshold, p85_threshold), already selected on validation. Pass this for
    ANY split that isn't the one doing model selection (i.e. test, or a --eval-only re-check) -- the
    thresholds are held fixed and simply applied, never re-swept against that split's own labels. Leave
    None only for the validation split during training, where sweeping IS the legitimate model-selection
    step (see tune_threshold()'s docstring).
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
    attn_threshold, p85_threshold = fixed_thresholds if fixed_thresholds is not None else (None, None)
    attn_metrics = tune_threshold(np.array(attn_scores), labels, threshold=attn_threshold)
    p85_metrics = tune_threshold(np.array(p85_scores), labels, threshold=p85_threshold)
    return attn_metrics, p85_metrics


# =====================================================================
# 6. MAIN
# =====================================================================

def load_subject_ids(manifest_csv: str) -> List[str]:
    df = pd.read_csv(manifest_csv)
    return df["subject_id"].astype(str).tolist()


def log_split_metrics(logger, split_name: str, attn_metrics: Dict[str, float], p85_metrics: Dict[str, float]) -> None:
    # optimal_threshold is printed explicitly for both methods -- attn and p85 are thresholded
    # independently (each gets its own value, whether swept fresh on validation or held fixed from
    # a prior validation sweep for test/eval-only), so an identical F1 between the two is NOT evidence
    # they secretly shared a threshold; this makes that verifiable directly from the log instead of
    # requiring anyone to trust that claim.
    logger.info(
        f"  [{split_name}] Attn: F1={attn_metrics['subject_macro_f1']:.4f} AUC={attn_metrics['roc_auc']:.4f} "
        f"thr={attn_metrics['optimal_threshold']:.2f} "
        f"Acc={attn_metrics['subject_accuracy']:.4f} Sens={attn_metrics['subject_sensitivity']:.4f} "
        f"Spec={attn_metrics['subject_specificity']:.4f} | "
        f"p85 (same frozen probs): F1={p85_metrics['subject_macro_f1']:.4f} AUC={p85_metrics['roc_auc']:.4f} "
        f"thr={p85_metrics['optimal_threshold']:.2f}"
    )


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    probe = build_frozen_probe(args, device, logger)
    logger.info(f"Loaded frozen probe from {args.probe_checkpoint} (head_type={args.head_type}); probe parameters are NOT trained here.")

    best_model_path = Path(args.checkpoint_dir) / args.checkpoint_filename if args.checkpoint_dir else Path(args.checkpoint_filename)

    if args.eval_only:
        if not args.resume_checkpoint:
            raise ValueError("--eval-only requires --resume-checkpoint (a checkpoint saved by a prior training run of this script).")
        if not args.val_manifest and not args.test_manifest:
            raise ValueError("--eval-only requires at least one of --val-manifest / --test-manifest to evaluate against.")

        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)

        # Same principle as build_frozen_probe(): the checkpoint's OWN saved architecture (if
        # present) is the source of truth, not whatever --attn-hidden-dim happens to be on this
        # invocation's command line -- --eval-only is commonly a separate process/CLI invocation
        # from the one that trained the checkpoint, so there's no guarantee the flags still match.
        if "attn_hidden_dim" in ckpt:
            hidden_dim = ckpt["attn_hidden_dim"]
            if hidden_dim != args.attn_hidden_dim:
                logger.warning(
                    f"--attn-hidden-dim ({args.attn_hidden_dim}) does not match the checkpoint's own "
                    f"saved attn_hidden_dim ({hidden_dim}) -- using the checkpoint's value."
                )
        else:
            hidden_dim = args.attn_hidden_dim
            logger.warning(
                f"--resume-checkpoint ({args.resume_checkpoint}) has no saved attn_hidden_dim metadata "
                f"-- it predates that being saved. Falling back to --attn-hidden-dim ({hidden_dim}); "
                f"if that doesn't match what this checkpoint was actually trained with, load_state_dict "
                f"will fail with a shape-mismatch error below."
            )

        attn_head = AttentionPoolingHead(
            num_patches=args.num_patches, emb_dim=args.cbra_dim,
            hidden_dim=hidden_dim, dropout=args.attn_dropout,
        ).to(device)
        attn_head.load_state_dict(ckpt["attn_head_state_dict"])
        logger.info(f"Loaded attention-head weights from {args.resume_checkpoint} (epoch {ckpt.get('epoch', '?')}) -- evaluation only, no training.")

        # Thresholds must come from validation, never be re-swept on test (see tune_threshold()'s
        # docstring). Prefer a fresh sweep on --val-manifest if given (this invocation's own
        # legitimate model-selection split); otherwise fall back to whatever threshold the checkpoint
        # itself was saved with, from whichever validation run originally produced it.
        fixed_thresholds = (ckpt["attn_metrics"]["optimal_threshold"], ckpt["p85_metrics_same_epoch"]["optimal_threshold"])
        if args.val_manifest:
            val_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.val_manifest))
            val_attn_metrics, val_p85_metrics = run_epoch_eval(attn_head, probe, val_ds, device)
            log_split_metrics(logger, "VAL", val_attn_metrics, val_p85_metrics)
            fixed_thresholds = (val_attn_metrics["optimal_threshold"], val_p85_metrics["optimal_threshold"])
        if args.test_manifest:
            test_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.test_manifest))
            log_split_metrics(logger, "TEST", *run_epoch_eval(attn_head, probe, test_ds, device, fixed_thresholds=fixed_thresholds))
        return

    if not args.train_manifest or not args.val_manifest:
        raise ValueError("--train-manifest and --val-manifest are both required (fixed split only in this first version).")

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

    attn_head = AttentionPoolingHead(
        num_patches=args.num_patches, emb_dim=args.cbra_dim,
        hidden_dim=args.attn_hidden_dim, dropout=args.attn_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(attn_head.parameters(), lr=args.attn_lr, weight_decay=args.weight_decay)

    best_f1, best_auc = 0.0, 0.0
    patience_counter = 0

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
                    # Explicit architecture metadata, same rationale as p08b's probe checkpoint --
                    # --eval-only is commonly a separate invocation from the one that trained this
                    # checkpoint, so --attn-hidden-dim on that later command line can't be trusted to
                    # still match what was actually trained. See main()'s --eval-only branch.
                    "attn_hidden_dim": args.attn_hidden_dim,
                    "attn_dropout": args.attn_dropout,
                    "num_patches": args.num_patches,
                    "cbra_dim": args.cbra_dim,
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

    if args.test_manifest:
        # Reload the actual BEST saved checkpoint before scoring test -- attn_head in memory right now
        # is whatever the LAST epoch trained produced, which early stopping means is not necessarily
        # (and in this run's case, was not) the same as the best-F1/AUC checkpoint that got saved.
        ckpt = torch.load(best_model_path, map_location="cpu", weights_only=True)
        attn_head.load_state_dict(ckpt["attn_head_state_dict"])
        logger.info(f"Reloaded best checkpoint (epoch {ckpt['epoch']}) from {best_model_path} for held-out test scoring.")

        # Thresholds are the ones selected on validation AT that best epoch -- held fixed and applied
        # to test, never re-swept against test's own labels (see tune_threshold()'s docstring for why
        # that distinction matters: it's the difference between an honest test score and one that's
        # silently tuned against the labels it's being scored on).
        fixed_thresholds = (ckpt["attn_metrics"]["optimal_threshold"], ckpt["p85_metrics_same_epoch"]["optimal_threshold"])

        test_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.test_manifest))
        leaked_test = (set(train_ds.unique_subjects) | set(val_ds.unique_subjects)) & set(test_ds.unique_subjects)
        assert not leaked_test, f"[CRITICAL LEAKAGE] {len(leaked_test)} subject(s) in test AND (train or val): {leaked_test}"

        logger.info("=" * 125)
        logger.info(
            f"HELD-OUT TEST ({len(test_ds)} subjects) -- scored with the BEST checkpoint selected on "
            f"validation, using that SAME validation-selected threshold (attn={fixed_thresholds[0]:.2f}, "
            f"p85={fixed_thresholds[1]:.2f}), not re-tuned on test:"
        )
        log_split_metrics(logger, "TEST", *run_epoch_eval(attn_head, probe, test_ds, device, fixed_thresholds=fixed_thresholds))
        logger.info("=" * 125)


if __name__ == "__main__":
    main()
