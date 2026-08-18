"""
p16_gated_attention_embedding_mil.py

Option B of the attention-based MIL follow-up: attention directly over frozen CBraMod embeddings
(Ilse et al., 2018, "Attention-based Deep Multiple Instance Learning"), with NO intermediate
window-level probe at all. A subject's recording is pooled into ONE representation --
`pooled = sum(attn_weight_i * embedding_i)` -- and a freshly-trained classifier head maps that
pooled representation directly to the subject-level prediction. There is no per-window probability
anywhere in this pipeline; "window_prob" from Option A (p13) simply doesn't exist here.

--------------------------------------------------------------------------
How this differs from Option A (p13_attention_mil_pooling.py), concretely
--------------------------------------------------------------------------
Option A kept the p08b-trained window-level probe completely frozen and untouched, and only replaced
the p85 pooling of its scalar outputs with a learned attention-weighted sum -- the causal-preservation
investigation (p15, docs/sigma_band_causal_investigation.md Sec 6.5) found this results in a clean
"division of labor": the probe carries the validated sigma/spindle causal signal, and the gate learned
something largely orthogonal (a stage/informativeness axis). That division is only possible because
there's a separate, already-informative scalar channel (window_prob) for the causal signal to live in
independently of whatever the gate itself learns.

Option B removes that separate channel entirely. Attention operates on raw embeddings and a NEW head
is trained jointly with the gate, purely from the subject-level (weak) supervision signal -- there is
no equivalent of "window_prob" that could carry a signal independently of what the attention weights
themselves end up encoding. Whether the same sigma-band mechanism gets rediscovered here, gets diluted
across a higher-capacity model, or gets replaced by something else entirely is an open, testable
question this script's checkpoint enables answering (via an embedding-based reinterpretation of
p14/p15's methodology) -- not addressed by this script itself.

--------------------------------------------------------------------------
Capacity / overfitting risk -- read before choosing defaults
--------------------------------------------------------------------------
This model has substantially more trainable parameters than Option A's gate (~780K vs. ~396K at
these defaults -- verified by direct computation, not eyeballed: gated attention needs TWO
in_features -> attn_hidden_dim projections (V and U) where Option A's gate needed only one, PLUS an
entirely new classifier head that Option A didn't need at all, since it reused the frozen probe's).
attn_hidden_dim defaults to 32 here specifically -- smaller than Option A's 64 -- as a partial
mitigation; even so this is roughly 2x Option A's parameter count. Fit against the same ~150-200
subject-level labels this cohort has, that is a much tighter capacity/data ratio than Option A's
already-tight one (which itself showed a real, visible overfitting shape -- val AUC peaking early,
then drifting down while train loss kept falling). Expect this to need MORE aggressive
regularization (dropout, weight decay)
and/or smaller hidden dims than Option A's defaults to avoid worse overfitting, not the same ones.

Usage:
    python p16_gated_attention_embedding_mil.py \
        --cache-dir /data/eeg_study/cache \
        --train-manifest /data/eeg_study/train_manifest.csv \
        --val-manifest /data/eeg_study/val_manifest.csv \
        --test-manifest /data/eeg_study/test_manifest.csv \
        --epochs 40 --head-lr 1e-3 --dropout 0.3
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
    GatedAttentionMIL,
    add_log_filename_argument,
    is_checkpoint_improvement,
    load_subject_ids,
    seed_everything,
    setup_cache_cli_parser,
    setup_training_cli_parser,
)


# =====================================================================
# 1. CLI
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = setup_training_cli_parser(
        description="Attention-MIL Option B: gated attention directly over frozen CBraMod embeddings, "
                    "with a jointly-trained subject-level head -- no intermediate window-level probe."
    )

    setup_cache_cli_parser(parser)

    model_group = parser.add_argument_group("Gated-Attention MIL Model")
    model_group.add_argument("--attn-hidden-dim", type=int, default=32, help="Hidden dim of the V/U gated-attention projections (smaller default than Option A -- see module docstring on capacity risk)")
    model_group.add_argument("--head-hidden-dim", type=int, default=64, help="Hidden dim of the subject-level classifier head")
    model_group.add_argument(
        "--subjects-per-step", type=int, default=8,
        help="Number of subjects (bags) whose losses get averaged before each optimizer.step() -- "
             "gradient accumulation, same rationale as p13's flag of the same name."
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--test-manifest", type=str, default=None)
    eval_group.add_argument("--eval-only", action="store_true")
    eval_group.add_argument("--resume-checkpoint", type=str, default=None)

    add_log_filename_argument(parser, __file__)

    args = parser.parse_args()
    return args


# =====================================================================
# 3. THRESHOLD TUNING (same pattern as p13 -- fixed threshold for test/eval-only, swept for val)
# =====================================================================

def tune_threshold(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
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
# 4. TRAIN / EVAL LOOPS
# =====================================================================

def run_epoch_train(
    model: GatedAttentionMIL, dataset: CachedFeatureSubjectDataset, optimizer: torch.optim.Optimizer,
    device: torch.device, subjects_per_step: int, subject_order: np.ndarray,
) -> float:
    model.train()
    total_loss, n_subjects = 0.0, 0
    optimizer.zero_grad()
    for i, subj_idx in enumerate(subject_order):
        bag_feats, label, _subject_id, _stages, _indices = dataset[subj_idx]
        bag_feats = bag_feats.to(device).float()
        label = label.to(device).long()

        logits, _attn_weights = model(bag_feats)
        loss = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

        (loss / subjects_per_step).backward()
        n_subjects += 1
        if (i + 1) % subjects_per_step == 0 or (i + 1) == len(subject_order):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / max(n_subjects, 1)


@torch.no_grad()
def run_epoch_eval(
    model: GatedAttentionMIL, dataset: CachedFeatureSubjectDataset, device: torch.device,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    scores, labels = [], []
    for subj_idx in range(len(dataset)):
        bag_feats, label, _subject_id, _stages, _indices = dataset[subj_idx]
        bag_feats = bag_feats.to(device).float()
        logits, _attn_weights = model(bag_feats)
        scores.append(torch.softmax(logits, dim=0)[1].item())
        labels.append(int(label.item()))
    return tune_threshold(np.array(scores), np.array(labels), threshold=fixed_threshold)


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    best_model_path = Path(args.checkpoint_dir) / args.checkpoint_filename if args.checkpoint_dir else Path(args.checkpoint_filename)

    if args.eval_only:
        if not args.resume_checkpoint:
            raise ValueError("--eval-only requires --resume-checkpoint.")
        if not args.val_manifest and not args.test_manifest:
            raise ValueError("--eval-only requires at least one of --val-manifest / --test-manifest.")

        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        attn_hidden_dim = ckpt.get("attn_hidden_dim", args.attn_hidden_dim)
        head_hidden_dim = ckpt.get("head_hidden_dim", args.head_hidden_dim)
        head_type = ckpt.get("head_type", args.head_type)
        if "attn_hidden_dim" not in ckpt:
            logger.warning(
                f"--resume-checkpoint has no saved architecture metadata -- falling back to CLI flags "
                f"(attn_hidden_dim={attn_hidden_dim}, head_hidden_dim={head_hidden_dim}, head_type={head_type}); "
                f"load_state_dict will fail below if any of these is wrong."
            )
        model = GatedAttentionMIL(
            num_patches=args.num_patches, emb_dim=args.cbra_dim, attn_hidden_dim=attn_hidden_dim,
            head_hidden_dim=head_hidden_dim, dropout=args.dropout, num_classes=args.num_classes,
            head_type=head_type,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded model from {args.resume_checkpoint} (epoch {ckpt.get('epoch', '?')}) -- evaluation only, no training.")

        fixed_threshold = ckpt.get("optimal_threshold")
        if args.val_manifest:
            val_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.val_manifest))
            val_metrics = run_epoch_eval(model, val_ds, device)
            logger.info(f"  [VAL]  F1={val_metrics['subject_macro_f1']:.4f} AUC={val_metrics['roc_auc']:.4f} thr={val_metrics['optimal_threshold']:.2f}")
            fixed_threshold = val_metrics["optimal_threshold"]
        if args.test_manifest:
            test_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.test_manifest))
            test_metrics = run_epoch_eval(model, test_ds, device, fixed_threshold=fixed_threshold)
            logger.info(f"  [TEST] F1={test_metrics['subject_macro_f1']:.4f} AUC={test_metrics['roc_auc']:.4f} thr={test_metrics['optimal_threshold']:.2f}")
        return

    if not args.train_manifest or not args.val_manifest:
        raise ValueError("--train-manifest and --val-manifest are both required (fixed split only in this first version).")

    train_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.train_manifest))
    val_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.val_manifest))
    leaked = set(train_ds.unique_subjects) & set(val_ds.unique_subjects)
    assert not leaked, f"[CRITICAL LEAKAGE] {len(leaked)} subject(s) in both train and val: {leaked}"
    logger.info(f"✓ [Leak Check Passed] Train: {len(train_ds)} subjects | Val: {len(val_ds)} subjects.")

    model = GatedAttentionMIL(
        num_patches=args.num_patches, emb_dim=args.cbra_dim, attn_hidden_dim=args.attn_hidden_dim,
        head_hidden_dim=args.head_hidden_dim, dropout=args.dropout, num_classes=args.num_classes,
        head_type=args.head_type,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"GatedAttentionMIL: {n_params:,} trainable parameters (attn_hidden_dim={args.attn_hidden_dim}, "
        f"head_hidden_dim={args.head_hidden_dim}, head_type={args.head_type})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)

    best_f1, best_auc = 0.0, 0.0
    patience_counter = 0
    logger.info(f"Starting Option B Training ({args.epochs} epochs max | subjects/step: {args.subjects_per_step})")
    logger.info("=" * 125)

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        subject_order = rng.permutation(len(train_ds))
        train_loss = run_epoch_train(model, train_ds, optimizer, device, args.subjects_per_step, subject_order)
        val_metrics = run_epoch_eval(model, val_ds, device)
        elapsed = time.time() - t0

        log_str = (
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.2f}s) | Train Loss: {train_loss:.4f} | "
            f"Val F1={val_metrics['subject_macro_f1']:.4f} AUC={val_metrics['roc_auc']:.4f}"
        )

        if is_checkpoint_improvement(val_metrics["subject_macro_f1"], val_metrics["roc_auc"], best_f1, best_auc):
            best_f1, best_auc = val_metrics["subject_macro_f1"], val_metrics["roc_auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "attn_hidden_dim": args.attn_hidden_dim,
                    "head_hidden_dim": args.head_hidden_dim,
                    "head_type": args.head_type,
                    "num_patches": args.num_patches,
                    "cbra_dim": args.cbra_dim,
                    "num_classes": args.num_classes,
                    "best_macro_f1": best_f1,
                    "best_auc": best_auc,
                    "optimal_threshold": val_metrics["optimal_threshold"],
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )
            log_str += " --> [BEST MODEL SAVED]"
        else:
            patience_counter += 1
            log_str += f" | EarlyStop: {patience_counter}/{args.patience}"

        logger.info(log_str)
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info("=" * 125)
    logger.info(f"Training Complete. Best Subject Macro F1: {best_f1:.4f} | Best AUC: {best_auc:.4f}")

    if args.test_manifest:
        ckpt = torch.load(best_model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Reloaded best checkpoint (epoch {ckpt['epoch']}) from {best_model_path} for held-out test scoring.")

        test_ds = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.test_manifest))
        leaked_test = (set(train_ds.unique_subjects) | set(val_ds.unique_subjects)) & set(test_ds.unique_subjects)
        assert not leaked_test, f"[CRITICAL LEAKAGE] {len(leaked_test)} subject(s) in test AND (train or val): {leaked_test}"

        test_metrics = run_epoch_eval(model, test_ds, device, fixed_threshold=ckpt["optimal_threshold"])
        logger.info("=" * 125)
        logger.info(
            f"HELD-OUT TEST ({len(test_ds)} subjects) -- scored with the BEST checkpoint, using its "
            f"validation-selected threshold ({ckpt['optimal_threshold']:.2f}), not re-tuned on test:"
        )
        logger.info(f"  F1={test_metrics['subject_macro_f1']:.4f} AUC={test_metrics['roc_auc']:.4f} "
                    f"Acc={test_metrics['subject_accuracy']:.4f} Sens={test_metrics['subject_sensitivity']:.4f} "
                    f"Spec={test_metrics['subject_specificity']:.4f}")
        logger.info("=" * 125)


if __name__ == "__main__":
    main()
