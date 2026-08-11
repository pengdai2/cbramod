"""
p09g_key_subject_feature_summary.py

A different question from p09f_morphology_score_correlation.py's within-
subject correlation ("does this feature predict THIS window's score, for
the SAME subject"). This asks a between-subject question instead: does a
subject's OVERALL feature level (e.g. mean sigma_relpower across all their
windows) relate to whether that subject was classified correctly, and how
confidently -- i.e. do misclassified subjects, or low-confidence correct
subjects, look systematically different from high-confidence correct ones
at the level of "how much of this feature does this subject have overall,"
as opposed to "does this feature vary window-to-window with the score
within one subject."

Reads the window-level CSV already produced by
p09f_morphology_score_correlation.py (no re-run of model inference
needed) and a p09d_subject_confidence_report.py --output-json report,
and reports per-subject mean/median feature values alongside that
subject's outcome/confidence label (FP/FN/highest_conf/lowest_conf).

With only a handful of subjects (as flagged by p09d) this is meant to be
eyeballed, not run through a formal significance test -- there isn't
enough n for one to mean much.

Usage:
  python p09g_key_subject_feature_summary.py \
      --morphology-csv morphology_score_correlation.csv \
      --subjects-json key_subjects.json \
      --stage N2
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-subject feature summary for p09d-flagged key subjects.")
    parser.add_argument(
        "--morphology-csv", type=str, required=True,
        help="Path to a morphology_score_correlation.csv from p09f_morphology_score_correlation.py."
    )
    parser.add_argument(
        "--subjects-json", type=str, required=True,
        help="Path to a p09d_subject_confidence_report.py --output-json report."
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Restrict to one stage (e.g. 'N2', the more reliable one per the within-subject "
             "correlation results). Default: use all stages present for each subject."
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Where to save the summary table. Default: alongside --morphology-csv."
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


def main():
    args = parse_cli_args()

    df = pd.read_csv(args.morphology_csv)
    labels = label_subjects(Path(args.subjects_json))

    df = df[df["subject_id"].isin(labels.keys())].copy()
    if df.empty:
        raise ValueError(
            f"None of --subjects-json's subject_ids were found in {args.morphology_csv} -- make sure "
            "they're from the same pool this CSV was built from (e.g. both validation, or both test)."
        )

    missing = set(labels.keys()) - set(df["subject_id"].unique())
    if missing:
        print(f"  [Warning] {len(missing)} subject(s) from --subjects-json not found in this CSV: {sorted(missing)}")

    if args.stage:
        df = df[df["stage"] == args.stage]
        if df.empty:
            raise ValueError(f"No rows left after restricting to --stage {args.stage}.")

    feature_cols = [c for c in df.columns if c not in
                     ("subject_id", "ground_truth", "raw_epoch_index", "stage", "probability")]

    rows = []
    for subj_id, group in df.groupby("subject_id"):
        row = {
            "subject_id": subj_id,
            "label": labels.get(subj_id, "UNKNOWN"),
            "ground_truth": group["ground_truth"].iloc[0],
            "n_windows": len(group),
            "mean_probability": group["probability"].mean(),
        }
        for col in feature_cols:
            row[f"mean_{col}"] = group[col].mean()
            row[f"median_{col}"] = group[col].median()
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    stage_note = f" (stage={args.stage} only)" if args.stage else " (all stages)"
    print(f"\nPer-subject feature summary{stage_note}:")
    print(summary.to_string(index=False))

    out_path = Path(args.output_csv) if args.output_csv else Path(args.morphology_csv).with_name(
        f"key_subjects_feature_summary{'_' + args.stage if args.stage else ''}.csv"
    )
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
