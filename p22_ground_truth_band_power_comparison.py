"""
p22_ground_truth_band_power_comparison.py

The direct, model-free version of "does sigma/delta/etc. actually differ between patients and
controls" -- everything reported so far in this investigation (p09f/p09k) correlates the MODEL's
predicted probability against band power, which is a claim about the model's learned decision rule,
not a claim about whether the raw recorded data actually separates the two groups. This script
answers the model-independent question directly: reads an ALREADY-SAVED p09k (or p09f) output CSV --
no new model inference, embeddings, or checkpoints needed -- and compares each band's power directly
against ground_truth, at TWO distinct levels:

  1. BETWEEN-subject: is the group difference a broad, cohort-wide shift, or driven by a handful of
     outlier subjects? Answered by comparing subject-level mean/median (each computed across ONE
     subject's own windows first) between the two ground-truth groups -- if mean, median, p25, AND
     p75 all move in the same direction by comparable magnitude, that's evidence of a broad shift
     across most subjects, not a few extreme ones skewing the mean.
  2. WITHIN-subject: for a TYPICAL subject, is the group difference itself spread broadly across
     ALL of their own windows, or concentrated in a minority "fat tail" of their own windows (while
     the bulk of their recording looks similar to the other group)? Answered by computing each
     subject's OWN window-level percentiles (p10/p25/median/p75/p90/p95/p99) first, then averaging
     those percentiles across subjects within each ground-truth group -- mirroring exactly the
     shape-comparison p21 already does for the model's predicted probability, but here applied to
     the raw band power itself.

These are genuinely separate questions -- a clean between-subject shift (1) does not by itself tell
you whether that shift is broad or tail-like within any given subject's own recording (2).

--------------------------------------------------------------------------
Why subject-level summaries, not pooled windows
--------------------------------------------------------------------------
Comparing every window from every subject pooled together conflates within-subject and
between-subject variance -- the same ecological-fallacy trap flagged throughout this investigation.
Both analyses above compute a per-subject summary FIRST, then compare across subjects -- never a raw
pooled-window comparison.

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

PERCENTILES = [10, 25, 50, 75, 90, 95, 99]


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
        description="Direct, model-free comparison of band power / morphology event counts between "
                    "ground-truth patients and controls, from an already-saved p09k/p09f output CSV "
                    "-- no new model inference needed. Reports both a between-subject comparison "
                    "(broad cohort-wide shift vs. outlier-driven) and a within-subject one (broad vs. "
                    "fat-tailed shape inside a typical subject's own recording).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--morphology-csv", action="append", required=True, dest="morphology_csvs",
        help="Path to a p09k (or p09f) output CSV. Repeat this flag to combine multiple runs (e.g. "
             "val + test) for a larger subject sample."
    )
    parser.add_argument("--output-csv", type=str, default="subject_level_band_power_by_ground_truth.csv")
    return parser.parse_args()


def build_subject_level_summary(windows: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    One row per subject: mean, median, AND the full percentile set (p10-p99) of each feature,
    computed across THAT SUBJECT's own windows -- named aggregation (not the old dict-of-lists +
    multi-index-flatten style) so percentile columns get clean, explicit names directly.
    """
    agg_kwargs = {"ground_truth": pd.NamedAgg(column="ground_truth", aggfunc="first")}
    for col in feature_cols:
        agg_kwargs[f"{col}_mean"] = pd.NamedAgg(column=col, aggfunc="mean")
        agg_kwargs[f"{col}_median"] = pd.NamedAgg(column=col, aggfunc="median")
        for p in PERCENTILES:
            agg_kwargs[f"{col}_p{p}"] = pd.NamedAgg(column=col, aggfunc=lambda x, p=p: x.quantile(p / 100.0))
    return windows.groupby("subject_id").agg(**agg_kwargs).reset_index()


def report_between_subject(subject_df: pd.DataFrame, feature_cols: List[str]) -> None:
    print("\n" + "=" * 88)
    print("1) BETWEEN-SUBJECT: broad cohort-wide shift, or a few outlier subjects?")
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
                _u_stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
            except ValueError:
                p_value = float("nan")

            print(f"\n  {summary_col}:")
            print(f"    patients (n={len(pos)}): mean={pos.mean():.4f} median={pos.median():.4f} p25={pos.quantile(.25):.4f} p75={pos.quantile(.75):.4f}")
            print(f"    controls (n={len(neg)}): mean={neg.mean():.4f} median={neg.median():.4f} p25={neg.quantile(.25):.4f} p75={neg.quantile(.75):.4f}")
            print(f"    Spearman(ground_truth, {stat}) = {r:+.4f}  |  Mann-Whitney U p-value = {p_value:.4f}")

    print(
        "\n  A meaningful, consistent (mean AND median) separation with a small Mann-Whitney p-value is "
        "direct, model-independent evidence the raw data itself separates the groups on that feature -- "
        "not just that the model learned to use it as a decision lever."
    )


def report_within_subject_shape(subject_df: pd.DataFrame, feature_cols: List[str]) -> None:
    print("\n" + "=" * 88)
    print("2) WITHIN-SUBJECT: is a TYPICAL subject's own window-level distribution broadly shifted,")
    print("   or is the group difference concentrated in a minority 'fat tail' of their own windows?")
    print("=" * 88)
    print(
        "  Each subject's OWN window-level percentiles are computed first, then averaged across "
        "subjects within each group. If patient and control percentiles track closely at the low "
        "end but diverge sharply only at p90+, that's a fat tail. If the gap is roughly consistent "
        "across ALL percentiles (p10 through p99), that's a broad shift within a typical subject's "
        "own recording, not a rare-event pattern. Discrete YASA counts (n_spindles/n_slow_waves) may "
        "look degenerate here (many zero-valued percentiles) simply because they're sparse integer "
        "counts, not continuous power -- read those with that caveat."
    )
    pctl_cols_template = [f"p{p}" for p in PERCENTILES]
    for col in feature_cols:
        cols = [f"{col}_{p}" for p in pctl_cols_template]
        if not all(c in subject_df.columns for c in cols):
            continue
        print(f"\n  {col}:")
        for gt in (1, 0):
            label_name = "positive/patient" if gt == 1 else "negative/control"
            sub = subject_df[subject_df["ground_truth"] == gt]
            if len(sub) == 0:
                continue
            means = sub[cols].mean()
            formatted = "  ".join(f"{p}={means[c]:.4f}" for p, c in zip(pctl_cols_template, cols))
            print(f"    ground_truth={gt} ({label_name}, n={len(sub)}): {formatted}")


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

    subject_df = build_subject_level_summary(windows, feature_cols)

    print(f"\nSubject-level summary: {len(subject_df)} subjects (from {windows['subject_id'].nunique()} in the raw CSV(s))")

    output_path = Path(args.output_csv)
    subject_df.to_csv(output_path, index=False)
    print(f"Saved subject-level summary to {output_path}")

    n_pos = int((subject_df["ground_truth"] == 1).sum())
    n_neg = int((subject_df["ground_truth"] == 0).sum())
    print(f"Ground truth: {n_pos} patients, {n_neg} controls")

    report_between_subject(subject_df, feature_cols)
    report_within_subject_shape(subject_df, feature_cols)


if __name__ == "__main__":
    main()
