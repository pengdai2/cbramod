"""
p09d_subject_confidence_report.py

Post-processes a `subject_predictions_<strategy>.csv` export written by
p09_clinical_inference.py's analyze_subject_results() to surface:

  1. Misclassified subjects, tagged FP (ground_truth=0, prediction=1) or
     FN (ground_truth=1, prediction=0).
  2. The correctly classified P and N subject(s) with the HIGHEST confidence.
  3. The correctly classified P and N subject(s) with the LOWEST confidence
     (i.e. the most borderline correct calls).

For binary-classification runs, analyze_subject_results() now bakes an
"outcome" (TP/TN/FP/FN) and "confidence" (|pooled_score - eval_t|, using the
exact decision threshold that produced the "prediction" column) column
directly into the CSV -- this script uses those columns when present, since
they're guaranteed consistent with "prediction" (no risk of a mismatched
threshold skewing the ranking). For older CSVs exported before that change
(missing "outcome"/"confidence"), it falls back to deriving them here from
--threshold, which you should then set to whatever "Checkpoint Threshold"
p09_clinical_inference.py printed for that strategy at the time.

Usage:
  python p09d_subject_confidence_report.py --csv subject_predictions_p85_score.csv
  python p09d_subject_confidence_report.py --csv subject_predictions_p85_score.csv --top-n 5 --output-csv annotated.csv
  # Older CSV missing outcome/confidence columns:
  python p09d_subject_confidence_report.py --csv old_subject_predictions.csv --threshold 0.42
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify misclassified and most/least-confident correctly-classified subjects "
                    "from a p09_clinical_inference.py subject_predictions_<strategy>.csv export."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to subject_predictions_<strategy>.csv")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fallback decision threshold, only used to derive outcome/confidence when the CSV "
             "predates analyze_subject_results() baking them in directly. Should match the "
             "'Checkpoint Threshold' p09_clinical_inference.py printed for this strategy at the "
             "time; ignored (with a notice) if the CSV already has outcome/confidence columns."
    )
    parser.add_argument(
        "--top-n", type=int, default=1,
        help="How many highest/lowest-confidence correctly-classified subjects to report per class (default: 1)."
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional path to save the full table annotated with outcome/confidence columns."
    )
    return parser.parse_args()


def annotate(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Derives outcome (TP/TN/FP/FN) and confidence (distance from threshold), for CSVs that don't already have them."""
    df = df.copy()
    df["ground_truth"] = df["ground_truth"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["pooled_score"] = df["pooled_score"].astype(float)

    is_correct = df["ground_truth"] == df["prediction"]
    df["outcome"] = [
        ("TP" if gt == 1 else "TN") if correct else ("FP" if pred == 1 else "FN")
        for gt, pred, correct in zip(df["ground_truth"], df["prediction"], is_correct)
    ]
    df["confidence"] = (df["pooled_score"] - threshold).abs()
    return df


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def report(df: pd.DataFrame, top_n: int) -> None:
    # 1. Misclassified subjects (FP/FN)
    misclassified = df[df["outcome"].isin(["FP", "FN"])].sort_values("confidence", ascending=False)
    print_section(f"1) MISCLASSIFIED SUBJECTS ({len(misclassified)} total)")
    if misclassified.empty:
        print("  (none)")
    else:
        for _, row in misclassified.iterrows():
            print(
                f"  [{row['outcome']}] {row['subject_id']}: pooled_score={row['pooled_score']:.4f}, "
                f"ground_truth={row['ground_truth']}, prediction={row['prediction']}"
            )

    # 2 & 3. Highest/lowest confidence correctly-classified subjects, per class
    for class_label, outcome_tag in [("P", "TP"), ("N", "TN")]:
        correct = df[df["outcome"] == outcome_tag].sort_values("confidence", ascending=False)

        print_section(f"2) HIGHEST-CONFIDENCE CORRECT [{class_label}] SUBJECTS (top {top_n})")
        if correct.empty:
            print("  (none)")
        else:
            for _, row in correct.head(top_n).iterrows():
                print(f"  {row['subject_id']}: pooled_score={row['pooled_score']:.4f}, confidence={row['confidence']:.4f}")

        print_section(f"3) LOWEST-CONFIDENCE CORRECT [{class_label}] SUBJECTS (top {top_n}, most borderline)")
        if correct.empty:
            print("  (none)")
        else:
            for _, row in correct.tail(top_n).sort_values("confidence").iterrows():
                print(f"  {row['subject_id']}: pooled_score={row['pooled_score']:.4f}, confidence={row['confidence']:.4f}")


def main():
    args = parse_cli_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"subject_id", "ground_truth", "pooled_score", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing expected column(s) {sorted(missing)}. Expected the "
            "binary-classification schema written by p09_clinical_inference.py's "
            "analyze_subject_results() (subject_id, ground_truth, pooled_score, prediction). "
            "Multi-class exports store pooled_score as a per-class list and aren't supported here."
        )

    if {"outcome", "confidence"}.issubset(df.columns):
        print(f"Using outcome/confidence columns already baked into '{csv_path}' by p09_clinical_inference.py.")
    else:
        print(
            f"'{csv_path}' predates baked-in outcome/confidence columns -- deriving them here from "
            f"--threshold={args.threshold}. Make sure this matches the checkpoint threshold that "
            f"actually produced the 'prediction' column, or the confidence ranking will be off."
        )
        df = annotate(df, args.threshold)

    report(df, args.top_n)

    if args.output_csv:
        out_path = Path(args.output_csv)
        df.to_csv(out_path, index=False)
        print(f"\nFull annotated table saved to: {out_path}")


if __name__ == "__main__":
    main()
