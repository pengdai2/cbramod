"""
p26_merge_feature_caches.py

Merges two p08a-produced feature caches (one per cohort) into a single combined
cache usable by p08b/p13/p16/p20/etc. via the ordinary single --cache-dir CLI --
no downstream code changes needed, since CachedFeatureSubjectDataset already
just loads whatever single .pt file it's pointed at.

Why this can't be "just point training at both caches": each cache's own
`labels` tensor was baked in from whichever manifest was passed to
p08a_extract_features.py at extraction time, and grins1/grins2 use INCOMPATIBLE
label encodings (grins1: control=0/patient=1; grins2: control=0/bipolar=1/scz=2/
unlabeled=-1 -- see p25_build_combined_manifest.py's docstring for the same issue
at the manifest level). Naively concatenating both caches' feats/labels would
silently mix two different label meanings.

This script instead treats the combined manifest (p25's output) as the single
source of truth for (a) which subjects to keep -- so grins2's excluded control/
scz/unlabeled subjects are dropped even though their windows are still sitting
in grins2's raw cache -- and (b) what each kept subject's label actually is,
completely REPLACING each cache's own `labels` tensor rather than trusting it.
The embeddings/stages/indices arrays are untouched (extraction doesn't depend on
labels at all), just filtered and relabeled.

Caveat this script can't verify on its own: both source caches should have been
extracted with the same p08a settings (--filter-stage, --sfreq, --num-channels)
-- if grins1's cache kept different sleep stages than grins2's, merging would
combine windows extracted under different filtering, not a like-for-like pool.
Check this against however you invoked p08a_extract_features.py for each cohort
before trusting the merged cache.

Usage:
  python p26_merge_feature_caches.py \
      --grins1-cache /data/grins1_cache/cached_master_embeddings.pt \
      --grins2-cache /data/grins2_cache/cached_master_embeddings.pt \
      --combined-manifest analysis/manifest/combined-3class/master_manifest.csv \
      --output-cache analysis/manifest/combined-3class/cached_master_embeddings.pt
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


def load_and_filter(pt_path: Path, subject_to_label: dict, cohort_name: str) -> dict:
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    missing_keys = [k for k in ("feats", "labels", "subject_ids", "stages", "indices") if k not in data]
    if missing_keys:
        raise KeyError(f"{pt_path} is missing key(s) {missing_keys} -- not a p08a-produced cache?")

    subject_ids = list(data["subject_ids"])
    keep_mask = torch.tensor([sid in subject_to_label for sid in subject_ids], dtype=torch.bool)
    n_total, n_kept = len(subject_ids), int(keep_mask.sum())
    print(f"  [{cohort_name}] Keeping {n_kept}/{n_total} windows "
          f"({len(set(s for s in subject_ids if s in subject_to_label))} subjects) "
          f"per the combined manifest's subject list.")

    kept_subject_ids = [sid for sid in subject_ids if sid in subject_to_label]
    new_labels = torch.tensor([subject_to_label[sid] for sid in kept_subject_ids], dtype=data["labels"].dtype)

    return {
        "feats": data["feats"][keep_mask],
        "labels": new_labels,  # fully replaces the cache's own labels -- see module docstring
        "subject_ids": kept_subject_ids,
        "stages": [s for s, k in zip(data["stages"], keep_mask.tolist()) if k],
        "indices": [i for i, k in zip(data["indices"], keep_mask.tolist()) if k],
    }


def main():
    parser = argparse.ArgumentParser(description="Merge two cohorts' p08a feature caches into one, relabeled from a combined manifest.")
    parser.add_argument("--grins1-cache", type=str, required=True)
    parser.add_argument("--grins2-cache", type=str, required=True)
    parser.add_argument("--combined-manifest", type=str, required=True,
                         help="p25_build_combined_manifest.py's master_manifest.csv -- authoritative for subject inclusion and labels")
    parser.add_argument("--output-cache", type=str, required=True)
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.combined_manifest)
    subject_to_label = dict(zip(manifest_df["subject_id"], manifest_df["label"]))
    print(f"Loaded {len(subject_to_label)} subjects from combined manifest: {args.combined_manifest}")

    print("Filtering + relabeling each cohort's cache...")
    grins1_filtered = load_and_filter(Path(args.grins1_cache), subject_to_label, "grins1")
    grins2_filtered = load_and_filter(Path(args.grins2_cache), subject_to_label, "grins2")

    merged = {
        "feats": torch.cat([grins1_filtered["feats"], grins2_filtered["feats"]], dim=0),
        "labels": torch.cat([grins1_filtered["labels"], grins2_filtered["labels"]], dim=0),
        "subject_ids": grins1_filtered["subject_ids"] + grins2_filtered["subject_ids"],
        "stages": grins1_filtered["stages"] + grins2_filtered["stages"],
        "indices": grins1_filtered["indices"] + grins2_filtered["indices"],
    }

    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    n_windows = merged["feats"].shape[0]
    n_subjects = len(set(merged["subject_ids"]))
    class_counts = {}
    for lbl in merged["labels"].tolist():
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    print(f"\nSaved merged cache: {output_path}")
    print(f"  {n_windows} windows across {n_subjects} subjects")
    print(f"  Window-level label distribution: {dict(sorted(class_counts.items()))}")

    manifest_subject_count = manifest_df["subject_id"].nunique()
    if n_subjects != manifest_subject_count:
        missing = set(subject_to_label) - set(merged["subject_ids"])
        print(f"\n  [Warning] Merged cache has {n_subjects} subjects but the combined manifest lists "
              f"{manifest_subject_count}. Missing from the caches: {sorted(missing) if missing else '(none -- check for duplicates instead)'}")


if __name__ == "__main__":
    main()
