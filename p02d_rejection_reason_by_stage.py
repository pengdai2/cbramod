"""
p02d_rejection_reason_by_stage.py

Cohort-wide version of the specific question p02c_analyze_window_artifacts.py's per-subject stage
correlation test doesn't isolate: is the EXTREME_ARTIFACT rejection reason specifically (not "any
rejection reason") disproportionately concentrated in N3?

Motivation: p03's subject-level gatekeeping applies a stricter rejection-rate threshold to N3
specifically (10% vs 15% overall). N3 slow waves are naturally higher-amplitude than other stages'
activity -- if the window-level EXTREME_ARTIFACT check (150uV std ceiling, on top of the +-500uV hard
clip) is disproportionately triggered by genuine high-amplitude N3 content rather than real artifact,
that stricter N3 gate could be systematically penalizing subjects for having strong, real slow-wave
sleep rather than bad data -- the opposite of its intent. This is checkable directly from a sliced
cohort's metadata without needing to change or rerun slicing first: none of the recent p02 bug fixes
touched the window-level rejection thresholds or clip bounds, so an existing (pre-fix) sliced cohort
is equally valid evidence for this specific question.

Reuses p02c_analyze_window_artifacts.py's load_and_parse_metadata() (not reimplemented) to build one
big per-window DataFrame across every subject in --sliced-dir, then:
  1. Reports EXTREME_ARTIFACT rate specifically (not "any rejection reason") per stage.
  2. Chi-square test isolating EXTREME_ARTIFACT-vs-not against stage (distinct from p02c's per-subject
     "any noisy" test) -- both cohort-wide (all stages) and N3-vs-everything-else-pooled (the direct,
     simple comparison motivating this check).
  3. Same breakdown for FLATLINE_DETECTED and EMPTY_CHANNEL_DATA, for context/contrast -- if EXTREME_
     ARTIFACT is uniquely stage-associated while the other two aren't, that's further evidence for the
     amplitude-confound hypothesis specifically (the other two rejection reasons have no obvious reason
     to be stage-dependent, so if they show similar stage-skew, something else is going on -- e.g. a
     stage-labeling artifact -- rather than the specific EXTREME_ARTIFACT mechanism suspected here).

Usage:
  python p02d_rejection_reason_by_stage.py --sliced-dir /path/to/sliced/
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from p02c_analyze_window_artifacts import load_and_parse_metadata


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cohort-wide check: is EXTREME_ARTIFACT rejection specifically concentrated in N3?"
    )
    parser.add_argument("--sliced-dir", type=str, required=True, help="Directory containing *_meta.json files (searched recursively).")
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional path to save the full per-window pooled DataFrame for further analysis."
    )
    return parser.parse_args()


def chi_square_reason_vs_stage(df: pd.DataFrame, reason: str, label: str) -> None:
    """Chi-square test of {stage} x {this specific reason vs everything else}, cohort-wide."""
    is_this_reason = df["quality_status"] == reason
    ct = pd.crosstab(df["stage"], is_this_reason.map({True: label, False: f"NOT_{label}"}))
    ct = ct.loc[ct.sum(axis=1) > 0]

    print(f"\n--- {label} rate by stage (cohort-wide) ---")
    rates = is_this_reason.groupby(df["stage"]).mean() * 100.0
    counts = is_this_reason.groupby(df["stage"]).sum().astype(int)
    totals = df.groupby("stage").size()
    for stage in totals.index:
        print(f"  {stage:8s}: {rates.get(stage, 0.0):6.2f}%  ({counts.get(stage, 0)}/{totals[stage]} windows)")

    if ct.shape[0] > 1 and ct.shape[1] > 1 and ct.values.sum() > 0:
        chi2, p_val, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        sig = "STATISTICALLY SIGNIFICANT" if p_val < 0.05 else "not statistically significant"
        print(f"  Chi-square: chi2={chi2:.2f}, dof={dof}, p={p_val:.4e}, Cramer's V={cramers_v:.4f} ({sig})")
    else:
        print("  [WARN] Insufficient variance to run chi-square test.")


def n3_vs_rest_comparison(df: pd.DataFrame, reason: str, label: str) -> None:
    """The direct, simple comparison motivating this whole check: N3 alone vs every other stage pooled."""
    is_n3 = df["stage"] == "N3"
    is_this_reason = df["quality_status"] == reason

    n3_rate = is_this_reason[is_n3].mean() * 100.0 if is_n3.any() else float("nan")
    rest_rate = is_this_reason[~is_n3].mean() * 100.0 if (~is_n3).any() else float("nan")

    ct = pd.crosstab(is_n3.map({True: "N3", False: "NOT_N3"}), is_this_reason.map({True: label, False: f"NOT_{label}"}))
    print(f"\n--- N3 vs. everything else, pooled: {label} rate ---")
    print(f"  N3:          {n3_rate:6.2f}%  ({is_this_reason[is_n3].sum()}/{is_n3.sum()} windows)")
    print(f"  NOT N3:      {rest_rate:6.2f}%  ({is_this_reason[~is_n3].sum()}/{(~is_n3).sum()} windows)")
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p_val, dof, _ = chi2_contingency(ct)
        sig = "STATISTICALLY SIGNIFICANT" if p_val < 0.05 else "not statistically significant"
        print(f"  Chi-square (2x2): chi2={chi2:.2f}, p={p_val:.4e} ({sig})")


def main():
    args = parse_cli_args()
    sliced_dir = Path(args.sliced_dir).resolve()

    meta_files = sorted(sliced_dir.rglob("*_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.json files found under {sliced_dir}.")

    print(f"Loading {len(meta_files)} subject metadata file(s)...")
    frames: List[pd.DataFrame] = []
    for mf in meta_files:
        try:
            frames.append(load_and_parse_metadata(mf))
        except ValueError as e:
            print(f"  [Skip] {mf.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    print(f"Pooled {len(df)} windows across {df['subject_id'].nunique()} subjects.\n")

    print("=" * 88)
    print("QUALITY_STATUS BREAKDOWN, COHORT-WIDE (denominator = all windows, valid + rejected)")
    print("=" * 88)
    print(df["quality_status"].value_counts(normalize=True).mul(100).round(2).to_string())

    print("\n" + "=" * 88)
    print("PRIMARY CHECK: is EXTREME_ARTIFACT specifically concentrated in N3?")
    print("=" * 88)
    chi_square_reason_vs_stage(df, "EXTREME_ARTIFACT", "EXTREME_ARTIFACT")
    n3_vs_rest_comparison(df, "EXTREME_ARTIFACT", "EXTREME_ARTIFACT")

    print("\n" + "=" * 88)
    print("CONTRAST: same checks for FLATLINE_DETECTED and EMPTY_CHANNEL_DATA")
    print("(no obvious reason these should be stage-dependent -- if they show similar stage-skew to")
    print("EXTREME_ARTIFACT, something else is going on, e.g. a stage-labeling artifact, rather than")
    print("the amplitude-specific mechanism this check is actually looking for.)")
    print("=" * 88)
    for reason in ["FLATLINE_DETECTED", "EMPTY_CHANNEL_DATA"]:
        if (df["quality_status"] == reason).any():
            chi_square_reason_vs_stage(df, reason, reason)
        else:
            print(f"\n--- {reason}: 0 occurrences in this cohort, skipping ---")

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved pooled per-window DataFrame to: {args.output_csv}")


if __name__ == "__main__":
    main()
