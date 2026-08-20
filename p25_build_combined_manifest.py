"""
p25_build_combined_manifest.py

Builds a single combined master_manifest.csv (+ train/val/test splits) for a new
3-way classification task (control / patient[scz] / bipolar) out of two
independently-sliced, independently-manifested cohorts:

  - grins1 (control vs. patient[scz], 2-class): kept in full.
  - grins2 (control / bipolar / scz / unlabeled, 3-class): only raw_label=="bipolar"
    rows are kept. grins2's own control and scz groups are deliberately excluded --
    see docs/sigma_band_causal_investigation.md Section 14 for why (grins2's own
    control population is measurably shifted on sigma/spindles relative to grins1's,
    and grins2's scz signal is weaker/less reliable than grins1's or grins2's
    bipolar signal at current sample sizes).

Neither cohort's own numeric `label` column is reused -- grins1 encodes
control=0/patient=1, grins2 encodes its own control=0/bipolar=1/scz=2/unlabeled=-1,
and these are NOT compatible encodings. This script derives a single fresh 3-way
encoding from each row's `raw_label` string instead: control=0, patient=1 (scz),
bipolar=2.

Split membership is PRESERVED from each cohort's own master_manifest.csv `split`
column (not re-shuffled) -- so grins1 subjects keep exactly the train/val/test
assignment used for model[0]'s original results (direct comparability), and
grins2's bipolar subjects keep the split they were already assigned.

Because the two cohorts' raw .npy/.meta files live under two different sliced-data
roots, this script resolves every row's npy_path/meta_path to an ABSOLUTE path
(joining each cohort's own --*-data-dir against its manifest's relative paths)
rather than leaving them relative to a single shared --data-dir. cbramod_common's
PANSubjectEEGDataset._index_dataset() already handles absolute paths as-is
(`if not raw_npy_path.is_absolute() else raw_npy_path`), so downstream training
scripts can point --data-dir at anything (or omit it) once the combined manifest
itself carries absolute paths.

Usage:
  python p25_build_combined_manifest.py \
      --grins1-manifest analysis/manifest/grins1-earlobe/master_manifest.csv \
      --grins1-data-dir /opt/cbra_data/30s_sliced/grins1-earlobe/sliced \
      --grins2-manifest analysis/manifest/grins2-earlobe/master_manifest.csv \
      --grins2-data-dir /opt/cbra_data/30s_sliced/grins2-earlobe/sliced \
      --output-dir analysis/manifest/combined-3class

  (--*-data-dir is each cohort's sliced-data root -- the directory
  npy_path/meta_path in that cohort's own master_manifest.csv are relative to;
  adjust to wherever each cohort's *_windows.npy/*_meta.json files actually live.)
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

LABEL_MAP = {"control": 0, "patient": 1, "bipolar": 2}


def resolve_absolute_paths(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Joins npy_path/meta_path against `data_dir` and rewrites them as absolute paths,
    so a combined manifest pooling two different sliced-data roots doesn't need a single
    shared --data-dir at training time (see module docstring)."""
    df = df.copy()
    for col in ("npy_path", "meta_path"):
        df[col] = df[col].apply(lambda p: str((data_dir / p).resolve()) if not Path(p).is_absolute() else p)
    return df


def load_and_filter_cohort(
    manifest_csv: Path, data_dir: Path, source_cohort: str, keep_raw_labels: Optional[set]
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    required_cols = {"subject_id", "npy_path", "meta_path", "num_slices", "valid_slices",
                      "sampling_freq", "raw_label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_csv} is missing expected column(s): {sorted(missing)}")

    total = len(df)
    if keep_raw_labels is not None:
        excluded = df[~df["raw_label"].isin(keep_raw_labels)]
        if len(excluded) > 0:
            excluded_counts = excluded["raw_label"].value_counts().to_dict()
            print(f"  [{source_cohort}] Excluding {len(excluded)}/{total} subjects not in {sorted(keep_raw_labels)}: "
                  f"{excluded_counts}")
        df = df[df["raw_label"].isin(keep_raw_labels)]

    unmapped = set(df["raw_label"].unique()) - set(LABEL_MAP)
    if unmapped:
        raise ValueError(f"{manifest_csv} has raw_label value(s) with no entry in LABEL_MAP: {sorted(unmapped)}")

    df = resolve_absolute_paths(df, data_dir)
    df["label"] = df["raw_label"].map(LABEL_MAP)
    df["source_cohort"] = source_cohort
    return df[["subject_id", "npy_path", "meta_path", "num_slices", "valid_slices",
               "sampling_freq", "raw_label", "label", "split", "source_cohort"]]


def main():
    parser = argparse.ArgumentParser(
        description="Build a combined grins1+grins2 3-class (control/patient/bipolar) manifest, "
                     "excluding grins2's own control and scz groups."
    )
    parser.add_argument("--grins1-manifest", type=str, required=True, help="Path to grins1's master_manifest.csv")
    parser.add_argument("--grins1-data-dir", type=str, required=True,
                         help="Sliced-data root grins1's npy_path/meta_path are relative to")
    parser.add_argument("--grins2-manifest", type=str, required=True, help="Path to grins2's master_manifest.csv")
    parser.add_argument("--grins2-data-dir", type=str, required=True,
                         help="Sliced-data root grins2's npy_path/meta_path are relative to")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the combined manifests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and filtering cohorts...")
    grins1_df = load_and_filter_cohort(
        Path(args.grins1_manifest), Path(args.grins1_data_dir), "grins1", keep_raw_labels=None
    )
    grins2_df = load_and_filter_cohort(
        Path(args.grins2_manifest), Path(args.grins2_data_dir), "grins2", keep_raw_labels={"bipolar"}
    )

    dupes = set(grins1_df["subject_id"]) & set(grins2_df["subject_id"])
    if dupes:
        raise ValueError(
            f"subject_id collision between grins1 and grins2 manifests: {sorted(dupes)} -- "
            f"resolve before combining (this script does not auto-prefix, unlike p24's cross-cohort tool)."
        )

    combined_df = pd.concat([grins1_df, grins2_df], ignore_index=True)
    combined_df.to_csv(output_dir / "master_manifest.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = combined_df[combined_df["split"] == split]
        split_df.to_csv(output_dir / f"{split}_manifest.csv", index=False)

    print("\n==========================================================================")
    print("      COMBINED 3-CLASS MANIFEST SUMMARY (control=0, patient/scz=1, bipolar=2)")
    print("==========================================================================")
    print(f"Total subjects: {len(combined_df)}  (grins1: {len(grins1_df)}, grins2 bipolar: {len(grins2_df)})")
    print(f"Saved: {output_dir / 'master_manifest.csv'}")
    print("--------------------------------------------------------------------------")
    for split in ["train", "val", "test"]:
        sub_df = combined_df[combined_df["split"] == split]
        class_counts = sub_df["label"].value_counts().to_dict()
        dist_str = ", ".join(f"Class {k}: {class_counts.get(k, 0)}" for k in sorted(LABEL_MAP.values()))
        cohort_counts = sub_df["source_cohort"].value_counts().to_dict()
        print(f" - {split.upper():<5} Set: {len(sub_df):>3} Subj | Dist: [{dist_str}] | By cohort: {cohort_counts}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
