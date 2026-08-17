"""
p22_ground_truth_band_power_comparison.py

The direct, model-free version of "does sigma/delta/etc. actually differ between patients and
controls" -- everything reported so far in this investigation (p09f/p09k) correlates the MODEL's
predicted probability against band power, which is a claim about the model's learned decision rule,
not a claim about whether the raw recorded data actually separates the two groups. This script
answers the model-independent question directly: reads an ALREADY-SAVED p09k (or p09f) output CSV --
no new model inference, embeddings, or checkpoints needed -- and compares each band's power directly
against ground_truth.

--------------------------------------------------------------------------
Why subject-level summaries, not pooled windows
--------------------------------------------------------------------------
Comparing every window from every subject pooled together conflates within-subject and
between-subject variance -- the same ecological-fallacy trap flagged throughout this investigation.
The correct comparison here is: one summary value (mean, and separately median) per SUBJECT across
their own windows, then compare THAT distribution between the two ground-truth groups. That's what
this script does -- never a raw pooled-window comparison.

--------------------------------------------------------------------------
What a result here would mean
--------------------------------------------------------------------------
If patients show meaningfully lower subject-level sigma power than controls in the RAW data (no model
involved), that's about as clean, model-independent evidence for the spindle-deficit hypothesis as is
available. If the groups look similar in raw sigma power despite the model's causal sensitivity to it
(established via perturbation testing elsewhere), that would be a more surprising and important
finding -- it would mean the model learned to USE sigma as a decision lever without sigma actually
being what separates this specific cohort in reality, pointing toward a confound/artifact
interpretation rather than a genuine physiological one.

Usage:
    python p22_ground_truth_band_power_comparison.py \
        --morphology-csv val_ckpt/absolute_band_power_analysis.csv \
        --morphology-csv test_ckpt/absolute_band_power_analysis.csv \
        --output-csv subject_level_band_power_by_ground_truth.csv
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via plain rank + Pearson (no scipy.stats dependency needed)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct, model-free comparison of subject-level band power / morphology event "
                    "counts between ground-truth patients and controls, from an already-saved p09k/"
                    "p09f output CSV -- no new model inference needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--morphology-csv", action="append", required=True, dest="morphology_csvs",
        help="Path to a p09k (or p09f) output CSV. Repeat this flag to combine multiple runs (e.g. "
             "val + test) for a larger subject sample."
    )
    parser.add_argument("--output-csv", type=str, default="subject_level_band_power_by_ground_truth.csv")
    return parser.parse_args()


def main():
    args = parse_cli_args()

    frames = []
    for csv_path in args.morphology_csvs:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"--morphology-csv not found: {path}")
        df = pd.read_csv(path)
        if "ground_truth" not in df.columns or "subject_id" not in df.columns:
            raise ValueError(f"{path} is missing 'ground_truth'/'subject_id' -- not a p09f/p09k output CSV?")
        frames.append(df)
        print(f"Loaded {len(df)} window-level rows from {path}")

    windows = pd.concat(frames, ignore_index=True)

    band_cols = [c for c in windows.columns if c.endswith("_real_abspower")]
    yasa_cols = [c for c in ("n_spindles", "n_slow_waves") if c in windows.columns]
    feature_cols = band_cols + yasa_cols
    if not feature_cols:
        raise ValueError(f"No band-power or YASA columns found in the input CSV(s) -- columns present: {list(windows.columns)}")

    # Subject-level summary FIRST -- one row per subject, mean/median across their own windows --
    # before any group comparison. This is what avoids the pooled-window ecological-fallacy trap.
    agg = {"ground_truth": "first"}
    for col in feature_cols:
        agg[col] = ["mean", "median"]
    subject_df = windows.groupby("subject_id").agg(agg)
    subject_df.columns = ["_".join(c).strip("_") if c[1] else c[0] for c in subject_df.columns]
    subject_df = subject_df.rename(columns={"ground_truth_first": "ground_truth"}).reset_index()

    n_dropped = windows["subject_id"].nunique() - len(subject_df)
    print(f"\nSubject-level summary: {len(subject_df)} subjects (from {windows['subject_id'].nunique()} in the raw CSV(s))")

    output_path = Path(args.output_csv)
    subject_df.to_csv(output_path, index=False)
    print(f"Saved subject-level summary to {output_path}\n")

    n_pos = int((subject_df["ground_truth"] == 1).sum())
    n_neg = int((subject_df["ground_truth"] == 0).sum())
    print(f"Ground truth: {n_pos} patients, {n_neg} controls\n")

    print("=" * 88)
    print("DIRECT, MODEL-FREE COMPARISON: does ground truth actually separate raw band power / morphology counts?")
    print("=" * 88)
    for col in feature_cols:
        for stat in ("mean", "median"):
            summary_col = f"{col}_{stat}"
            if summary_col not in subject_df.columns:
                continue
            pos = subject_df.loc[subject_df["ground_truth"] == 1, summary_col].dropna()
            neg = subject_df.loc[subject_df["ground_truth"] == 0, summary_col].dropna()
            if len(pos) < 2 or len(neg) < 2:
                continue

            r = spearman_corr(subject_df["ground_truth"].values, subject_df[summary_col].values)
            try:
                u_stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
            except ValueError:
                u_stat, p_value = float("nan"), float("nan")

            print(f"\n  {summary_col}:")
            print(f"    patients (n={len(pos)}): mean={pos.mean():.4f} median={pos.median():.4f} p25={pos.quantile(.25):.4f} p75={pos.quantile(.75):.4f}")
            print(f"    controls (n={len(neg)}): mean={neg.mean():.4f} median={neg.median():.4f} p25={neg.quantile(.25):.4f} p75={neg.quantile(.75):.4f}")
            print(f"    Spearman(ground_truth, {stat}) = {r:+.4f}  |  Mann-Whitney U p-value = {p_value:.4f}")

    print(
        "\nA meaningful, consistent (mean AND median) separation with a small Mann-Whitney p-value is "
        "direct, model-independent evidence the raw data itself separates the groups on that feature -- "
        "not just that the model learned to use it as a decision lever. Weak/inconsistent separation "
        "here despite a strong causal effect established elsewhere would be the more surprising, "
        "confound-suggestive result."
    )


if __name__ == "__main__":
    main()
