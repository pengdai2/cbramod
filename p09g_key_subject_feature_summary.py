"""
p09g_key_subject_feature_summary.py

Computes each subject's OWN within-subject correlation between window-level
probability and each spectral/morphology feature (the same quantity
p09f_morphology_score_correlation.py's "WITHIN-SUBJECT CORRELATION" section
computes internally per subject, but only ever reports folded into a
population mean/median/frac -- never exposed per subject). That per-subject
correlation is the actual "morphology score" for that subject: how strongly
(and in which direction) does this specific subject's own sigma/delta/etc.
power track their own window-to-window probability.

This lets you:
  1. Compare individual subjects' own correlations against each other and
     against the population's mean/median (computed here from the same
     CSV, across every subject in the pool -- not just the ones flagged by
     p09d, so "the statistical average" is a real reference, not a
     remembered number from an earlier run).
  2. Check whether a subject's own correlation strength for a given
     feature relates at all to that subject's subject-level prediction
     confidence -- i.e. do subjects with a strong, clean within-subject
     sigma relationship tend to be the ones the model is more (or less)
     confident about, computed across the whole pool for real statistical
     power rather than eyeballing 6 subjects.

Does NOT collapse window-level power to a per-subject mean (that was the
wrong tool for this question -- averaging away the window-to-window
variation throws out exactly the thing that correlates with probability).
Everything here operates on each subject's own within-subject correlation,
computed from their own window-level rows.

Subject-level confidence is derived here directly from the same CSV via
p85 percentile pooling (matching the p85_score strategy these results were
generated under) and a supplied --threshold -- not read from a separate
file, so this works standalone against just the morphology CSV.

Usage:
  python p09g_key_subject_feature_summary.py \
      --morphology-csv morphology_score_correlation.csv \
      --threshold 0.63 --stage N2 \
      --subjects-json key_subjects.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-subject within-subject correlation ('morphology score'), compared against the "
                    "population and against subject-level confidence."
    )
    parser.add_argument(
        "--morphology-csv", type=str, required=True,
        help="Path to a morphology_score_correlation.csv from p09f_morphology_score_correlation.py."
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Operating decision threshold for this checkpoint/strategy (the 'Checkpoint Threshold' "
             "printed by p09_clinical_inference.py) -- used only to derive each subject's confidence "
             "(|pooled_score - threshold|), not to recompute predictions."
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Restrict the correlation computation to one stage (e.g. 'N2', the more reliable one per "
             "the earlier stage-stratified results). Default: use all stages present for each subject. "
             "Pooled-score/confidence still use ALL of a subject's windows regardless of this flag, "
             "since that's what actually produced their real subject-level prediction."
    )
    parser.add_argument(
        "--min-windows", type=int, default=5,
        help="Minimum windows (after any --stage restriction) required to compute a subject's own "
             "correlation for a feature; below this it's reported as NaN rather than a noisy estimate."
    )
    parser.add_argument(
        "--subjects-json", type=str, default=None,
        help="Optional path to a p09d_subject_confidence_report.py --output-json report. If given, "
             "adds a 'label' column (FP/FN/TP_highest_conf/...) for those subject_ids so they're easy "
             "to pick out of the full-population table; every subject in --morphology-csv is still "
             "included and used for the population reference and the confidence-relationship check."
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Where to save the full per-subject table. Default: alongside --morphology-csv."
    )
    return parser.parse_args()


def label_subjects(subjects_json_path: Path) -> Dict[str, str]:
    """Maps subject_id -> a human-readable outcome/confidence label from a p09d --output-json report."""
    payload = json.load(open(subjects_json_path))
    labels: Dict[str, str] = {}

    for row in payload.get("misclassified", []):
        labels[row["subject_id"]] = row["outcome"]  # "FP" or "FN"

    for bucket, tag in [("highest_confidence", "highest_conf"), ("lowest_confidence", "lowest_conf")]:
        for cls_rows in payload.get(bucket, {}).values():
            for row in cls_rows:
                labels[row["subject_id"]] = f"{row['outcome']}_{tag}"

    return labels


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    args = parse_cli_args()

    df = pd.read_csv(args.morphology_csv)
    labels = label_subjects(Path(args.subjects_json)) if args.subjects_json else {}

    feature_cols = [c for c in df.columns if c not in
                     ("subject_id", "ground_truth", "raw_epoch_index", "stage", "probability")]

    rows = []
    for subj_id, full_group in df.groupby("subject_id"):
        # Pooled score / confidence always use ALL of this subject's windows, regardless of --stage --
        # that's what actually determined their real subject-level prediction, so restricting it to one
        # stage here would compute a confidence that doesn't match the one the model actually produced.
        pooled_score_p85 = float(np.percentile(full_group["probability"].values, 85))
        confidence = abs(pooled_score_p85 - args.threshold)

        stage_group = full_group[full_group["stage"] == args.stage] if args.stage else full_group

        row = {
            "subject_id": subj_id,
            "label": labels.get(subj_id, ""),
            "ground_truth": int(full_group["ground_truth"].iloc[0]),
            "n_windows": len(stage_group),
            "pooled_score_p85": pooled_score_p85,
            "confidence": confidence,
        }
        for col in feature_cols:
            if len(stage_group) >= args.min_windows and stage_group[col].std() > 0:
                row[f"r_{col}"] = spearman_corr(stage_group["probability"].values, stage_group[col].values)
            else:
                row[f"r_{col}"] = float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    r_cols = [c for c in summary.columns if c.startswith("r_")]

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    stage_note = f" (correlations computed on stage={args.stage} only)" if args.stage else " (all stages)"
    print(f"\n1) POPULATION REFERENCE across all {len(summary)} subjects{stage_note} -- 'the statistical average':")
    pop_ref = summary[r_cols].agg(["mean", "median", "count"]).T
    pop_ref.columns = ["mean_r", "median_r", "n_subjects_with_valid_r"]
    print(pop_ref.to_string())

    if labels:
        print(f"\n2) FLAGGED (p09d) SUBJECTS, for direct comparison against the population reference above:")
        flagged = summary[summary["label"] != ""].sort_values("label")
        print(flagged.to_string(index=False))

    print(f"\n3) FULL PER-SUBJECT TABLE{stage_note}:")
    print(summary.sort_values("confidence", ascending=False).to_string(index=False))

    print("\n" + "=" * 88)
    print("BONUS: does a subject's OWN correlation strength for a feature relate to their confidence?")
    print("=" * 88)
    for col in r_cols:
        valid = summary.dropna(subset=[col, "confidence"])
        r = spearman_corr(valid[col].values, valid["confidence"].values)
        print(f"  Spearman({col}, confidence) = {r:+.4f}  (n_subjects={len(valid)})")

    out_path = Path(args.output_csv) if args.output_csv else Path(args.morphology_csv).with_name(
        f"per_subject_morphology_score{'_' + args.stage if args.stage else ''}.csv"
    )
    summary.to_csv(out_path, index=False)
    print(f"\nSaved full per-subject table to: {out_path}")


if __name__ == "__main__":
    main()
