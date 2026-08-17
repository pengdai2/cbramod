"""
p21_model0_confidence_stratification.py

First step toward the "bootstrap from model[0]" variant of Option C discussed after Option C's
failure: stratifies every subject in train/val by whether the naive, already-validated probe
(model[0] -- p08b's collectively-trained window-level probe, pooled via its own calibrated
threshold) gets that subject's LABEL right, and how confidently. This determines which subjects are
trustworthy enough to derive per-window pseudo-labels from (for training a hard-pseudo-labeling or
soft-distillation variant of Option C) and which aren't.

--------------------------------------------------------------------------
Why this has to come before any pseudo-labeling scheme is built
--------------------------------------------------------------------------
The whole point of bootstrapping from model[0] instead of training from scratch (as Option C's Stage
1 did) is that model[0]'s per-window scores are more trustworthy where its AGGREGATE (subject-level)
call is independently verified to be right. Where model[0] confidently gets a subject wrong, there's
nothing trustworthy to derive a pseudo-label from for that subject -- using it anyway would just
inject a different kind of label noise. This script produces the actual counts needed to decide how
much of the training set falls into each bucket, rather than assuming.

--------------------------------------------------------------------------
Category definitions
--------------------------------------------------------------------------
For each subject, using ONE pooling strategy's score (default: p85_score) and model[0]'s OWN
already-calibrated threshold for that strategy (saved in the checkpoint at training time -- never
re-derived here, exactly the same "threshold comes from validation, not from the split you're
scoring" discipline used throughout this investigation):

    signed_margin = (pooled_score - threshold)      if ground_truth == 1 (positive/patient)
                  = (threshold - pooled_score)       if ground_truth == 0 (negative/control)

signed_margin > 0 means correctly classified (regardless of magnitude); its absolute value is how
confidently, in either direction. Four categories per subject, crossed with ground_truth (8 total):

    confident_correct       : signed_margin >=  confidence_margin
    marginal_correct        : 0 <= signed_margin <  confidence_margin
    marginal_misclassified  : -confidence_margin <= signed_margin < 0
    confident_misclassified : signed_margin < -confidence_margin

--confidence-margin has NO principled default -- the right value depends on the actual scale of this
probe's pooled scores relative to its own threshold (which has varied a lot across runs in this
investigation, e.g. thresholds anywhere from 0.01 to 0.7). This script always prints the full
signed_margin distribution (by ground_truth) BEFORE applying any --confidence-margin cutoff, so it can
be chosen by looking at the actual numbers rather than guessed blind.

--------------------------------------------------------------------------
Window-level tail characterization (proving/quantifying the "fat tail")
--------------------------------------------------------------------------
p09c's window-level probability histograms suggested that for ANY subject, the bulk of windows sit
below threshold -- what distinguishes a patient is a minority "fat tail" of windows extending above
it, not a uniform upward shift of every window. This script makes that precise and checkable rather
than relying on a recalled visual impression:

  - tail_fraction: fraction of a subject's own windows scoring >= threshold. If the fat-tail
    description is right, this should be small but consistently positive for patients, and smaller
    (near zero) for controls.
  - window_prob percentiles (p10/p25/median/p75/p90/p95/p99): characterizes the SHAPE, not just one
    summary number -- if patients and controls have similar low/median percentiles but diverge
    sharply at p90+, that's the fat tail showing up precisely, rather than a uniform shift.
  - mean_naive_ce_loss: the per-window cross-entropy loss each window would contribute if trained
    under p08b's naive collective-assumption label (every window of a subject labeled with that
    subject's own ground truth). This is a direct, checkable prediction from the fat-tail hypothesis:
    if most of a patient's windows score low but are labeled 1 (naive assumption), that's a real,
    persistent mismatch and should show up as an ELEVATED mean_naive_ce_loss for positive subjects
    relative to negative ones -- computed here directly from the already-trained checkpoint, no
    retraining needed to check this.

Usage:
    python p21_model0_confidence_stratification.py \
        --cache-dir /data/eeg_study/cache \
        --train-manifest train_manifest.csv --val-manifest val_manifest.csv \
        --probe-checkpoint checkpoints-probe-linear/cbramod_ckpt.pt \
        --pooling-strategy p85_score --confidence-margin 0.1
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from cbramod_common import CachedFeatureSubjectDataset, compute_pooled_scores, setup_common_cli_parser
from cbramod_utils import setup_logger
from p13_attention_mil_pooling import build_frozen_probe


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratifies subjects by model[0]'s subject-level correctness/confidence, to "
                    "inform a bootstrap-from-model[0] variant of Option C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_common_cli_parser(parser)

    cache_group = parser.add_argument_group("Cache Controls")
    cache_group.add_argument("--cache-dir", type=str, required=True)
    cache_group.add_argument("--master-cache-name", type=str, default="cached_master_embeddings.pt")

    ckpt_group = parser.add_argument_group("Model[0] (frozen probe)")
    ckpt_group.add_argument("--probe-checkpoint", type=str, required=True, help="model[0] -- the already-validated probe checkpoint to stratify against")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-manifest", type=str, required=True)
    data_group.add_argument("--val-manifest", type=str, required=True)
    data_group.add_argument(
        "--test-manifest", type=str, default=None,
        help="Optional -- reported for information only (e.g. sanity-checking this script), NEVER "
             "to be used to inform any training decision downstream of this analysis."
    )

    strat_group = parser.add_argument_group("Stratification")
    strat_group.add_argument(
        "--pooling-strategy", type=str, default="p85_score",
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"],
        help="Which pooling strategy's score/threshold to stratify by."
    )
    strat_group.add_argument(
        "--confidence-margin", type=float, default=0.1,
        help="No principled default -- see module docstring. Choose after looking at the printed "
             "signed_margin distribution, not blind."
    )
    strat_group.add_argument("--output-csv", type=str, default="model0_confidence_stratification.csv")
    strat_group.add_argument("--log-filename", type=str, default=Path(__file__).stem + ".log")

    return parser.parse_args()


def load_subject_ids(manifest_csv: str) -> List[str]:
    df = pd.read_csv(manifest_csv)
    return df["subject_id"].astype(str).tolist()


def compute_naive_ce_loss(window_probs: np.ndarray, ground_truth: int, eps: float = 1e-7) -> float:
    """
    Mean per-window cross-entropy loss against p08b's naive collective-assumption label (every
    window of a subject labeled with that subject's own ground truth) -- computed directly from an
    already-trained probe's window_probs, no retraining needed. This is the direct, checkable
    prediction from the fat-tail hypothesis: if the bulk of a positive subject's windows score low
    (matching controls) despite being labeled 1, that mismatch shows up as an elevated value here.
    """
    p = np.clip(window_probs, eps, 1 - eps)
    per_window_loss = -np.log(p) if ground_truth == 1 else -np.log(1 - p)
    return float(per_window_loss.mean())


def stratify_split(
    dataset: CachedFeatureSubjectDataset, probe, device: torch.device,
    pooling_strategy: str, threshold: float, confidence_margin: float, split_name: str,
) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for subj_idx in range(len(dataset)):
            bag_feats, label, subject_id, _stages, _indices = dataset[subj_idx]
            bag_feats = bag_feats.to(device).float()
            logits = probe(bag_feats)
            window_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pooled_score = float(compute_pooled_scores(window_probs, method=pooling_strategy))

            ground_truth = int(label.item())
            signed_margin = (pooled_score - threshold) if ground_truth == 1 else (threshold - pooled_score)
            is_correct = signed_margin > 0
            is_confident = abs(signed_margin) >= confidence_margin
            category = f"{'confident' if is_confident else 'marginal'}_{'correct' if is_correct else 'misclassified'}"

            wp_pctl = np.percentile(window_probs, [10, 25, 50, 75, 90, 95, 99])
            tail_fraction = float((window_probs >= threshold).mean())
            mean_naive_ce_loss = compute_naive_ce_loss(window_probs, ground_truth)

            rows.append({
                "split": split_name,
                "subject_id": subject_id,
                "ground_truth": ground_truth,
                "n_windows": bag_feats.shape[0],
                "pooled_score": pooled_score,
                "threshold": threshold,
                "signed_margin": signed_margin,
                "is_correct": is_correct,
                "is_confident": is_confident,
                "category": category,
                "tail_fraction": tail_fraction,
                "mean_naive_ce_loss": mean_naive_ce_loss,
                "window_prob_p10": wp_pctl[0],
                "window_prob_p25": wp_pctl[1],
                "window_prob_median": wp_pctl[2],
                "window_prob_p75": wp_pctl[3],
                "window_prob_p90": wp_pctl[4],
                "window_prob_p95": wp_pctl[5],
                "window_prob_p99": wp_pctl[6],
            })
    return pd.DataFrame(rows)


def report_split(df: pd.DataFrame, split_name: str, confidence_margin: float) -> None:
    print("\n" + "=" * 88)
    print(f"SPLIT = {split_name} ({len(df)} subjects)")
    print("=" * 88)

    print("\n--- signed_margin distribution BY GROUND TRUTH (look at this before trusting --confidence-margin) ---")
    for gt in (1, 0):
        sub = df[df["ground_truth"] == gt]["signed_margin"]
        if len(sub) == 0:
            continue
        label_name = "positive/patient" if gt == 1 else "negative/control"
        percentiles = np.percentile(sub, [0, 25, 50, 75, 100])
        print(
            f"  ground_truth={gt} ({label_name}, n={len(sub)}): "
            f"min={percentiles[0]:+.4f} p25={percentiles[1]:+.4f} median={percentiles[2]:+.4f} "
            f"p75={percentiles[3]:+.4f} max={percentiles[4]:+.4f}"
        )

    print(f"\n--- Category counts at --confidence-margin={confidence_margin} ---")
    for gt in (1, 0):
        label_name = "positive/patient" if gt == 1 else "negative/control"
        sub = df[df["ground_truth"] == gt]
        print(f"  ground_truth={gt} ({label_name}, n={len(sub)}):")
        for cat in ["confident_correct", "marginal_correct", "marginal_misclassified", "confident_misclassified"]:
            n = int((sub["category"] == cat).sum())
            print(f"    {cat:<24}: {n}")

    print("\n--- Window-level shape: percentiles BY GROUND TRUTH (mean across subjects) ---")
    print("If patients and controls track closely at low percentiles but diverge sharply at p90+,")
    print("that's the fat tail showing up precisely, not a uniform upward shift.")
    pctl_cols = ["window_prob_p10", "window_prob_p25", "window_prob_median", "window_prob_p75", "window_prob_p90", "window_prob_p95", "window_prob_p99"]
    for gt in (1, 0):
        label_name = "positive/patient" if gt == 1 else "negative/control"
        sub = df[df["ground_truth"] == gt]
        if len(sub) == 0:
            continue
        means = sub[pctl_cols].mean()
        print(
            f"  ground_truth={gt} ({label_name}): "
            f"p10={means['window_prob_p10']:.4f} p25={means['window_prob_p25']:.4f} "
            f"median={means['window_prob_median']:.4f} p75={means['window_prob_p75']:.4f} "
            f"p90={means['window_prob_p90']:.4f} p95={means['window_prob_p95']:.4f} p99={means['window_prob_p99']:.4f}"
        )

    print("\n--- Tail fraction (fraction of a subject's own windows scoring >= threshold) BY GROUND TRUTH ---")
    for gt in (1, 0):
        label_name = "positive/patient" if gt == 1 else "negative/control"
        sub = df[df["ground_truth"] == gt]["tail_fraction"]
        if len(sub) == 0:
            continue
        print(f"  ground_truth={gt} ({label_name}): mean={sub.mean():.4f} median={sub.median():.4f}")

    print("\n--- Mean per-window naive-label CE loss BY GROUND TRUTH (the elevated-loss prediction) ---")
    print("If the fat-tail hypothesis is right, positive subjects should show a MEANINGFULLY HIGHER")
    print("value here than negative subjects -- most of their windows score low but are labeled 1.")
    for gt in (1, 0):
        label_name = "positive/patient" if gt == 1 else "negative/control"
        sub = df[df["ground_truth"] == gt]["mean_naive_ce_loss"]
        if len(sub) == 0:
            continue
        print(f"  ground_truth={gt} ({label_name}): mean={sub.mean():.4f} median={sub.median():.4f}")


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=True)
    thresholds = ckpt.get("optimal_thresholds", {})
    if args.pooling_strategy not in thresholds:
        raise ValueError(
            f"--probe-checkpoint has no calibrated threshold for '{args.pooling_strategy}' in its "
            f"optimal_thresholds ({list(thresholds.keys())}) -- was this checkpoint saved by p08b/p20?"
        )
    threshold = thresholds[args.pooling_strategy]
    logger.info(f"Using model[0]'s own calibrated threshold for {args.pooling_strategy}: {threshold:.4f} (never re-derived here)")

    probe = build_frozen_probe(args, device, logger)
    logger.info(f"Loaded model[0] from {args.probe_checkpoint}")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    splits = {"train": args.train_manifest, "val": args.val_manifest}
    if args.test_manifest:
        splits["test (INFO ONLY -- do not use downstream)"] = args.test_manifest

    all_dfs = []
    for split_name, manifest_path in splits.items():
        dataset = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(manifest_path))
        df = stratify_split(dataset, probe, device, args.pooling_strategy, threshold, args.confidence_margin, split_name)
        all_dfs.append(df)
        report_split(df, split_name, args.confidence_margin)

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = Path(args.output_csv)
    combined.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(combined)} per-subject rows to {output_path}")


if __name__ == "__main__":
    main()
