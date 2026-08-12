"""
Combines multiple already-completed p09i_subject_level_perturbation_test.py output CSVs into a
single perturb_fraction correlation analysis, WITHOUT re-running anything.

Why this exists: p09i's --perturb-fraction sweep was added after some runs (e.g. a 100%-perturb run,
and a 50%-perturb run) had already been kicked off or completed. Each run's CSV comes from a
different script-version schema:

  1. Oldest (pre --perturb-fraction): no `n_windows_perturbed` or `perturb_fraction` column at all.
     Every window was implicitly perturbed -- fraction is always 1.0.
  2. Single-float --perturb-fraction (commit 582bb00): has `n_windows_perturbed`, but the fraction
     value itself was only used internally to compute it and was never written to the row.
  3. Current (comma-separated --perturb-fraction sweep): has `perturb_fraction` directly.

Re-running the completed/in-progress jobs just to get the new column populated would waste the time
already sunk into them, so this script instead reconstructs `perturb_fraction` per row using a
fallback chain and concatenates everything before handing off to the same reporting logic p09i uses
for a single run.

Usage:
    python3 p09l_combine_perturb_fraction_runs.py \
        --input-csv runs/100pct/subject_level_perturbation.csv \
        --input-csv runs/50pct/subject_level_perturbation.csv \
        --input-csv runs/25pct/subject_level_perturbation.csv \
        --band sigma \
        --output-csv combined_perturb_fraction.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from p09i_subject_level_perturbation_test import report_fraction_sweep


def derive_perturb_fraction(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Ensures df has a `perturb_fraction` column, deriving it from whatever the source schema
    actually recorded:
      - `perturb_fraction` column present -> use as-is (current schema).
      - `n_windows_perturbed` present (no `perturb_fraction`) -> n_windows_perturbed / n_windows
        (582bb00 single-float schema).
      - neither present -> assume 1.0; every window was perturbed (oldest schema), and backfill
        `n_windows_perturbed` = n_windows for consistency with newer rows.
    """
    df = df.copy()
    if "perturb_fraction" in df.columns:
        pass
    elif "n_windows_perturbed" in df.columns:
        df["perturb_fraction"] = df["n_windows_perturbed"] / df["n_windows"]
    else:
        df["perturb_fraction"] = 1.0
        df["n_windows_perturbed"] = df["n_windows"]
    df["source_run"] = source_label
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-csv", action="append", required=True, dest="input_csvs",
        help="Path to a p09i output CSV. Repeat this flag once per run to combine (any schema generation).",
    )
    parser.add_argument("--band", type=str, default="the band", help="Band label, for the report header only.")
    parser.add_argument("--output-csv", type=str, default="combined_perturb_fraction.csv")
    args = parser.parse_args()

    frames = []
    for csv_path in args.input_csvs:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"--input-csv not found: {path}")
        raw = pd.read_csv(path)
        if "n_windows" not in raw.columns:
            raise ValueError(f"{path} is missing required column 'n_windows' -- not a p09i output CSV?")
        frames.append(derive_perturb_fraction(raw, source_label=path.stem))
        print(f"Loaded {len(raw)} rows from {path} (fractions present: {sorted(frames[-1]['perturb_fraction'].unique())})")

    df = pd.concat(frames, ignore_index=True)

    dup_key = ["subject_id", "perturb_fraction"]
    dupes = df[df.duplicated(subset=dup_key, keep=False)]
    if not dupes.empty:
        print(
            f"\nWARNING: {len(dupes)} rows share a (subject_id, perturb_fraction) pair across input runs "
            f"-- if two of your input CSVs cover the same fraction for the same subject, keeping both "
            f"double-counts that subject at that fraction in the summary below.\n"
            f"Affected (subject_id, perturb_fraction, source_run):\n"
            f"{dupes[['subject_id', 'perturb_fraction', 'source_run']].to_string(index=False)}"
        )

    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False)
    print(f"\nCombined {len(df)} rows from {len(args.input_csvs)} run(s) into: {output_path}")

    perturb_fractions = sorted(df["perturb_fraction"].unique())
    report_fraction_sweep(df, perturb_fractions, band_label=args.band)


if __name__ == "__main__":
    main()
