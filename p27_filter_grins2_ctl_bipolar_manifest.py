"""
p27_filter_grins2_ctl_bipolar_manifest.py

Builds a control-vs-bipolar-only manifest (+ train/val/test splits) from grins2's own
master_manifest.csv, for training a fresh 2-class model within grins2 alone -- deliberately
NOT combined with grins1, to avoid the cohort confound documented in
docs/sigma_band_causal_investigation.md Section 15 (a combined 3-class model's "bipolar"
predictions turned out to be partly driven by cohort/acquisition identity rather than
diagnosis, since bipolar was the only grins2-sourced class). Training within grins2 alone
also sidesteps Section 14's grins2-vs-grins1 control-baseline-shift finding entirely --
this model never needs grins2's control population to be comparable to grins1's, since it's
never compared to grins1 at all.

Unlike p25_build_combined_manifest.py, this does NOT need to:
  - resolve npy_path/meta_path to absolute paths (single cohort, single data_dir, unchanged)
  - relabel from raw_label (grins2's own manifest already encodes control=0/bipolar=1, which
    is exactly the 2-class encoding this task wants)
  - merge or relabel any feature cache -- CachedFeatureSubjectDataset (see p08b) filters
    the existing full grins2 cache down to whichever subject_ids appear in the manifest
    passed as --train-manifest/--val-manifest, and reads labels from the CACHE itself, not
    the manifest. So the existing full-cohort grins2 cache can be reused as-is; only the
    manifest needs filtering.

grins2's own scz and unlabeled subjects are simply dropped (not relabeled, not retained under
another name) -- this is a narrower-scope model by design, usable only for control-vs-bipolar,
per the explicit rationale in the conversation this script came out of.

Usage:
  python p27_filter_grins2_ctl_bipolar_manifest.py \
      --grins2-manifest analysis/manifest/grins2-earlobe/master_manifest.csv \
      --output-dir analysis/manifest/grins2-ctl-bipolar
"""

import argparse
from pathlib import Path

import pandas as pd

KEEP_RAW_LABELS = {"control", "bipolar"}


def main():
    parser = argparse.ArgumentParser(
        description="Filter grins2's own manifest down to control+bipolar subjects only."
    )
    parser.add_argument("--grins2-manifest", type=str, required=True, help="Path to grins2's master_manifest.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the filtered manifests")
    args = parser.parse_args()

    df = pd.read_csv(args.grins2_manifest)
    required_cols = {"subject_id", "raw_label", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{args.grins2_manifest} is missing expected column(s): {sorted(missing)}")

    excluded = df[~df["raw_label"].isin(KEEP_RAW_LABELS)]
    if len(excluded) > 0:
        print(f"Excluding {len(excluded)}/{len(df)} subjects: {excluded['raw_label'].value_counts().to_dict()}")
    filtered = df[df["raw_label"].isin(KEEP_RAW_LABELS)].copy()

    # Sanity check: this task assumes grins2's own encoding is already control=0/bipolar=1 --
    # fail loudly rather than silently training a model whose labels mean something unexpected.
    label_by_raw = filtered.groupby("raw_label")["label"].unique().to_dict()
    if set(label_by_raw.get("control", [])) != {0} or set(label_by_raw.get("bipolar", [])) != {1}:
        raise ValueError(
            f"Expected raw_label 'control'->label 0 and 'bipolar'->label 1, found {label_by_raw}. "
            "This script assumes grins2's own manifest encoding; re-check before proceeding."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_dir / "master_manifest.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = filtered[filtered["split"] == split]
        split_df.to_csv(output_dir / f"{split}_manifest.csv", index=False)

    print(f"\nSaved filtered manifest(s) to: {output_dir}")
    print(f"Total: {len(filtered)} subjects ({dict(filtered['raw_label'].value_counts())})")
    for split in ["train", "val", "test"]:
        s = filtered[filtered["split"] == split]
        print(f" - {split.upper():<5}: {len(s):>3} subjects ({dict(s['raw_label'].value_counts())})")


if __name__ == "__main__":
    main()
