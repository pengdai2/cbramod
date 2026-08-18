"""
p09j_band_covariation_check.py

Tests an assumption baked into p09h/p09i's `preserve_total_energy` perturbation design: that
increasing one band's power necessarily comes at the expense of the others (a zero-sum trade-off).
That assumption was never verified against the real data -- it's possible the opposite is true, that
bands co-move (e.g. from broadband amplitude modulation: arousal, movement, impedance change), in
which case forcing them apart (or forcing them to stay perfectly fixed) is an unrealistic
counterfactual, not a clean one.

IMPORTANT CAVEAT: morphology_score_correlation.csv (p09f's output) stores RELATIVE power --
delta_relpower + theta_relpower + alpha_relpower + sigma_relpower + beta_relpower ~= 1.0 for every
window (compositional/ratio data). Correlations between components of a fixed-sum composition are
subject to a well-known statistical artifact ("spurious correlation from closure", Pearson 1897):
if delta's share is large, the others are mechanically squeezed in RATIO terms even if their real
ABSOLUTE (uV^2) power moves the same direction as delta's absolute power. So:
  - The relpower covariation this script computes is real and worth seeing, but a negative
    correlation among relpower columns does NOT by itself prove bands trade off in an absolute sense.
  - Section 1 below directly measures how close each window's relpower sum is to 1.0, to confirm/
    quantify how tightly the closure constraint actually binds in this data.
  - Section 2 computes both pooled and within-subject correlations of delta_relpower against each
    other band, and each other-band pair against each other -- if the four non-delta bands are
    POSITIVELY correlated with each other (while each is negatively correlated with delta), that's
    the signature of "one dominant component crowding out a shared remainder", consistent with
    closure. If they're not even correlated with each other, delta's negative correlation with each
    is a more genuine, independent pattern.
  - This script cannot settle the ABSOLUTE-power version of the question (does real uV^2 delta power
    rise together with real uV^2 theta/alpha/sigma/beta power across windows?) -- that needs power
    computed on the reconstructed real-uV signal (recoverable via p02's saved norm_mean_uv/
    norm_std_uv), which p09f never computed. If this relpower check is ambiguous or suggestive, the
    next step is adding an absolute-power variant to p09f rather than trusting this proxy further.

Usage:
  python p09j_band_covariation_check.py --morphology-csv morphology_score_correlation.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cbramod_stats import spearman_corr


BAND_COLS = ["delta_relpower", "theta_relpower", "alpha_relpower", "sigma_relpower", "beta_relpower"]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real covariation between band relative powers across windows.")
    parser.add_argument("--morphology-csv", type=str, required=True, help="Path to p09f's morphology_score_correlation.csv.")
    return parser.parse_args()


def report_pairwise(df: pd.DataFrame, cols, label: str):
    print(f"\n--- {label} (n_windows={len(df)}) ---")
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = spearman_corr(df[c1].values, df[c2].values)
            print(f"  Spearman({c1:20s}, {c2:20s}) = {r:+.4f}")


def main():
    args = parse_cli_args()
    df = pd.read_csv(args.morphology_csv)
    missing = [c for c in BAND_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"--morphology-csv is missing expected columns {missing}; is this a p09f output file?")

    print("=" * 88)
    print("1) HOW TIGHTLY DOES THE CLOSURE CONSTRAINT BIND? (sum of the 5 relpower columns per window)")
    print("=" * 88)
    rel_sum = df[BAND_COLS].sum(axis=1)
    print(f"  mean={rel_sum.mean():.4f}  std={rel_sum.std():.4f}  min={rel_sum.min():.4f}  max={rel_sum.max():.4f}")
    print(
        "  If this is tightly clustered near 1.0, the 5 relpower columns are a near-exact composition -- "
        "any one column moving up mechanically forces the sum of the rest down by about the same amount, "
        "REGARDLESS of what the real absolute power is doing. That alone can produce a negative correlation "
        "between a dominant band and the rest even if their real (uV^2) power co-moves."
    )

    print("\n" + "=" * 88)
    print("2) PAIRWISE COVARIATION AMONG RELATIVE-POWER COLUMNS")
    print("=" * 88)
    print(
        "Look specifically at whether the FOUR NON-DELTA bands correlate POSITIVELY with each other "
        "(while each correlates negatively with delta) -- that pattern (one dominant component vs. a "
        "co-moving shared remainder) is the closure signature. If the non-delta bands don't correlate "
        "with each other either, delta's negative relationship with each is more likely a genuine, "
        "independent pattern rather than an artifact of the ratio construction."
    )
    report_pairwise(df, BAND_COLS, "ALL WINDOWS, POOLED (between + within subject mixed)")

    print("\n--- WITHIN-SUBJECT (median across subjects, mirrors p09f's within-subject section) ---")
    per_subject_rs = {f"{c1}|{c2}": [] for i, c1 in enumerate(BAND_COLS) for c2 in BAND_COLS[i + 1:]}
    for _, g in df.groupby("subject_id"):
        if len(g) < 5:
            continue
        for i, c1 in enumerate(BAND_COLS):
            for c2 in BAND_COLS[i + 1:]:
                r = spearman_corr(g[c1].values, g[c2].values)
                if not np.isnan(r):
                    per_subject_rs[f"{c1}|{c2}"].append(r)
    for pair, rs in per_subject_rs.items():
        if rs:
            print(f"  {pair:45s}: median_r={np.median(rs):+.4f}  mean_r={np.mean(rs):+.4f}  (n_subjects={len(rs)})")

    print(
        "\nReminder: this whole script only speaks to RELATIVE power covariation, which is bounded by the "
        "closure constraint checked in section 1. It cannot confirm or rule out real absolute-power "
        "co-increase -- that needs band power computed on the reconstructed real-uV signal."
    )


if __name__ == "__main__":
    main()
