"""
p24_cross_cohort_relabel.py

Builds a p22-ready window-level CSV out of two DIFFERENT cohorts' p09k/p09f output CSVs, so
p22_ground_truth_band_power_comparison.py's existing (unmodified) report logic can compare a group
from one cohort directly against a group from another -- e.g. "grins2 bipolar patients vs. grins1
controls" -- rather than only ever comparing groups within the SAME cohort's own ground_truth column.

--------------------------------------------------------------------------
Why this check matters
--------------------------------------------------------------------------
A model trained on cohort A's patient/control distinction has its decision boundary calibrated
specifically against cohort A's OWN control population. If cohort B's control population turns out
to differ from cohort A's (different recruitment, site, age, medication-free-status, whatever), then
comparing cohort B's patients against cohort B's OWN controls answers a different question than
comparing cohort B's patients against the reference frame the model actually learned -- and the two
can disagree, sometimes sharply (see docs/sigma_band_causal_investigation.md's grins1/grins2 cross-
cohort investigation, where exactly this happened: grins2's controls turned out to have substantially
lower sigma/spindle counts than grins1's controls, which was largely responsible for an apparent
"reversed" sigma effect in grins2's bipolar group that mostly vanished once compared against grins1's
controls directly instead).

--------------------------------------------------------------------------
Why --reference-csv/--target-csv are repeatable
--------------------------------------------------------------------------
This is a purely model-free, descriptive comparison (means/medians/Mann-Whitney/Spearman on raw band
power) -- not a held-out model evaluation -- so there's no train/test leakage concern in using ALL of
a cohort's available subjects (train+val+test) as the reference population, not just whichever split
happened to already have a p09k run. More subjects means a materially more reliable reference
distribution, which matters most for exactly the small-n groups (e.g. 29 controls) this kind of check
tends to involve. Repeat the flag once per split's CSV; all rows across every repeated --reference-csv
(or --target-csv) matching that side's --*-ground-truth value are pooled into one group.

--------------------------------------------------------------------------
What this script does NOT do
--------------------------------------------------------------------------
No new statistics, no new report logic -- it only relabels and concatenates. All the actual
comparison (Mann-Whitney U, Spearman, within-subject percentile shape) still comes from running
p22_ground_truth_band_power_comparison.py, unmodified, on this script's output CSV.

Usage:
    # grins2 bipolar patients vs. ALL of grins1's own controls (train+val+test):
    python p24_cross_cohort_relabel.py \
        --reference-csv grins1/train_ckpt/absolute_band_power_analysis.csv \
        --reference-csv grins1/val_ckpt/absolute_band_power_analysis.csv \
        --reference-csv grins1/test_ckpt/absolute_band_power_analysis.csv \
        --reference-ground-truth 0 --reference-label grins1_control \
        --target-csv grins2/absolute_band_power_analysis-bpd_ctl.csv --target-ground-truth 1 --target-label grins2_bipolar \
        --output-csv cross_grins1ctrl_vs_grins2bpd.csv

    python p22_ground_truth_band_power_comparison.py --morphology-csv cross_grins1ctrl_vs_grins2bpd.csv

    # grins2's own controls vs. grins1's controls (the baseline-shift check itself):
    python p24_cross_cohort_relabel.py \
        --reference-csv grins1/train_ckpt/absolute_band_power_analysis.csv \
        --reference-csv grins1/val_ckpt/absolute_band_power_analysis.csv \
        --reference-csv grins1/test_ckpt/absolute_band_power_analysis.csv \
        --reference-ground-truth 0 --reference-label grins1_control \
        --target-csv grins2/absolute_band_power_analysis-bpd_ctl.csv --target-ground-truth 0 --target-label grins2_control \
        --output-csv cross_grins1ctrl_vs_grins2ctrl.csv
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabels and concatenates p09k/p09f output CSVs (potentially from different "
                    "cohorts/splits) into one p22-ready CSV, so a group from one cohort can be "
                    "compared directly against a group from another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference-csv", action="append", required=True, dest="reference_csvs",
        help="p09k/p09f output CSV for the REFERENCE group (printed as 'controls'/ground_truth=0 by "
             "p22). Repeatable -- e.g. once per split (train/val/test) -- rows from every repeated "
             "CSV matching --reference-ground-truth are pooled into one group."
    )
    parser.add_argument("--reference-ground-truth", type=int, required=True, choices=[0, 1], help="Which ground_truth value to keep from every --reference-csv.")
    parser.add_argument("--reference-label", type=str, required=True, help="Short tag identifying the reference group (e.g. 'grins1_control') -- prefixed onto subject_id and stored in a new source_label column.")
    parser.add_argument(
        "--target-csv", action="append", required=True, dest="target_csvs",
        help="p09k/p09f output CSV for the TARGET group (printed as 'patients'/ground_truth=1 by "
             "p22). Repeatable, same semantics as --reference-csv."
    )
    parser.add_argument("--target-ground-truth", type=int, required=True, choices=[0, 1], help="Which ground_truth value to keep from every --target-csv.")
    parser.add_argument("--target-label", type=str, required=True, help="Short tag identifying the target group (e.g. 'grins2_bipolar').")
    parser.add_argument("--output-csv", type=str, required=True, help="Where to write the combined, relabeled CSV.")
    return parser.parse_args()


def load_and_relabel(csv_paths: List[str], keep_ground_truth: int, label: str, new_ground_truth: int) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        missing = {"subject_id", "ground_truth"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing {missing} -- not a p09k/p09f output CSV?")

        df = df[df["ground_truth"] == keep_ground_truth].copy()
        if len(df) == 0:
            raise ValueError(f"{path} has no rows with ground_truth == {keep_ground_truth} -- wrong CSV or wrong value?")

        n_subjects = df["subject_id"].nunique()
        print(f"  {path}: kept {len(df)} window rows across {n_subjects} subjects (ground_truth=={keep_ground_truth})")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    n_total_subjects_before = combined["subject_id"].nunique()
    if n_total_subjects_before < sum(f["subject_id"].nunique() for f in frames):
        print(
            f"  [Warning] {label}: subject_id overlaps across the {len(csv_paths)} --*-csv inputs "
            f"(e.g. the same subject appearing in two splits) -- {n_total_subjects_before} unique "
            f"subject_ids from {sum(f['subject_id'].nunique() for f in frames)} total rows-with-a-"
            f"subject-id-count. Double check these are genuinely disjoint splits."
        )

    # Prefix subject_id with the cohort/group label -- guarantees no accidental cross-cohort ID
    # collision even when the two source cohorts are already known to use non-overlapping schemes,
    # and makes which row came from which source immediately visible without a separate lookup.
    combined["subject_id"] = label + "_" + combined["subject_id"].astype(str)
    combined["source_label"] = label
    combined["ground_truth"] = new_ground_truth
    print(f"  -> '{label}': {len(combined)} total window rows across {n_total_subjects_before} subjects, relabeled (ground_truth->{new_ground_truth})")
    return combined


def main():
    args = parse_cli_args()

    print("Reference group:")
    reference_df = load_and_relabel(args.reference_csvs, args.reference_ground_truth, args.reference_label, new_ground_truth=0)
    print("Target group:")
    target_df = load_and_relabel(args.target_csvs, args.target_ground_truth, args.target_label, new_ground_truth=1)

    ref_cols, tgt_cols = set(reference_df.columns), set(target_df.columns)
    only_in_reference = ref_cols - tgt_cols
    only_in_target = tgt_cols - ref_cols
    if only_in_reference:
        print(f"[Warning] Columns only in --reference-csv (will be NaN for target rows): {sorted(only_in_reference)}")
    if only_in_target:
        print(f"[Warning] Columns only in --target-csv (will be NaN for reference rows): {sorted(only_in_target)}")

    combined = pd.concat([reference_df, target_df], ignore_index=True, sort=False)

    output_path = Path(args.output_csv)
    combined.to_csv(output_path, index=False)
    print(
        f"\nSaved {len(combined)} window rows ({reference_df['subject_id'].nunique()} '{args.reference_label}' "
        f"subjects as ground_truth=0, {target_df['subject_id'].nunique()} '{args.target_label}' subjects as "
        f"ground_truth=1) to {output_path}"
    )
    print(f"Run: python p22_ground_truth_band_power_comparison.py --morphology-csv {output_path}")


if __name__ == "__main__":
    main()
