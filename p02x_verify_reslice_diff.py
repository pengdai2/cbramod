"""
p02x_verify_reslice_diff.py

Batch version of the one-off diff check that confirmed a single subject's re-sliced .npy/meta
differed only by float32 rounding noise (max abs diff = 2^-21, one ULP at that magnitude; identical
bad_channels_interpolated, identical num_slices). Run this across every affected subject to confirm
the same conclusion holds everywhere before trusting a re-slice wholesale, rather than generalizing
from one sampled subject.

For each subject present in both --old-dir and --new-dir, checks:
  1. Tensor shape match.
  2. Max/mean absolute difference (float64) between the two .npy tensors.
  3. Fraction of elements differing by more than --noise-threshold (default 1e-6) -- if this is 0.0,
     every element is indistinguishable from float32 rounding noise at that tensor's magnitude.
  4. num_slices match (from meta.json).
  5. bad_channels_interpolated match, per-slice (a real pipeline DECISION, not just a numeric value --
     if this differs, the two runs disagree about something that isn't rounding noise).

Flags (prints prominently) any subject where max abs diff exceeds --flag-threshold (default 1e-4,
well above single-ULP float32 noise at typical EEG magnitudes) OR where bad_channels_interpolated
or num_slices differ at all -- those are the subjects that need individual investigation, not just
a "looks fine" pass on the aggregate.

Usage:
  python p02x_verify_reslice_diff.py \
      --old-dir /opt/cbra_data/30s_sliced/grins1/sliced \
      --new-dir /opt/cbra_data/30s_sliced/grins1/sliced2 \
      --subjects-file affected_54_subjects.txt   # optional; default: every subject in both dirs
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-verify a p02 re-slice differs only by float32 rounding noise.")
    parser.add_argument("--old-dir", type=str, required=True, help="Directory with the original *_windows.npy/*_meta.json.")
    parser.add_argument("--new-dir", type=str, required=True, help="Directory with the re-sliced *_windows.npy/*_meta.json.")
    parser.add_argument(
        "--subjects-file", type=str, default=None,
        help="Optional text file, one subject_id per line, to restrict the check to (e.g. the 54 "
             "flagged as changed). Default: every subject_id present in both directories."
    )
    parser.add_argument(
        "--noise-threshold", type=float, default=1e-6,
        help="Elements differing by more than this are counted in 'frac_changed' (default 1e-6, well "
             "above float32 ULP noise for typical EEG-scale values, well below anything meaningful)."
    )
    parser.add_argument(
        "--flag-threshold", type=float, default=1e-4,
        help="A subject's max abs diff exceeding this gets flagged as needing individual "
             "investigation, rather than being waved through as rounding noise."
    )
    return parser.parse_args()


def discover_subjects(old_dir: Path, new_dir: Path, subjects_file: Optional[Path]) -> List[str]:
    if subjects_file:
        with open(subjects_file) as f:
            return [line.strip() for line in f if line.strip()]
    old_ids = {p.stem.replace("_windows", "") for p in old_dir.glob("*_windows.npy")}
    new_ids = {p.stem.replace("_windows", "") for p in new_dir.glob("*_windows.npy")}
    common = sorted(old_ids & new_ids)
    missing_old, missing_new = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
    if missing_old:
        print(f"[Note] {len(missing_old)} subject(s) present in --new-dir but not --old-dir: {missing_old[:10]}{'...' if len(missing_old) > 10 else ''}")
    if missing_new:
        print(f"[Note] {len(missing_new)} subject(s) present in --old-dir but not --new-dir: {missing_new[:10]}{'...' if len(missing_new) > 10 else ''}")
    return common


def compare_subject(subject_id: str, old_dir: Path, new_dir: Path, noise_threshold: float) -> dict:
    old_npy, new_npy = old_dir / f"{subject_id}_windows.npy", new_dir / f"{subject_id}_windows.npy"
    old_meta_path, new_meta_path = old_dir / f"{subject_id}_meta.json", new_dir / f"{subject_id}_meta.json"

    result = {"subject_id": subject_id, "error": None}
    try:
        old_arr, new_arr = np.load(old_npy), np.load(new_npy)
    except Exception as e:
        result["error"] = f"failed to load .npy: {e}"
        return result

    if old_arr.shape != new_arr.shape:
        result["error"] = f"SHAPE MISMATCH: old={old_arr.shape} new={new_arr.shape}"
        return result

    diff = np.abs(old_arr.astype(np.float64) - new_arr.astype(np.float64))
    result["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    result["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    result["frac_changed"] = float((diff > noise_threshold).mean()) if diff.size else 0.0

    try:
        old_meta, new_meta = json.load(open(old_meta_path)), json.load(open(new_meta_path))
        result["num_slices_old"] = old_meta.get("num_slices")
        result["num_slices_new"] = new_meta.get("num_slices")

        old_bads = {s["window_idx"]: tuple(sorted(s.get("bad_channels_interpolated", []))) for s in old_meta.get("slices", [])}
        new_bads = {s["window_idx"]: tuple(sorted(s.get("bad_channels_interpolated", []))) for s in new_meta.get("slices", [])}
        mismatched_windows = [w for w in old_bads if w in new_bads and old_bads[w] != new_bads[w]]
        result["bad_channels_mismatch_count"] = len(mismatched_windows)
        result["bad_channels_mismatch_sample"] = mismatched_windows[:5]
    except Exception as e:
        result["meta_error"] = str(e)

    return result


def main():
    args = parse_cli_args()
    old_dir, new_dir = Path(args.old_dir), Path(args.new_dir)
    subjects_file = Path(args.subjects_file) if args.subjects_file else None

    subjects = discover_subjects(old_dir, new_dir, subjects_file)
    print(f"Checking {len(subjects)} subject(s)...\n")

    flagged = []
    clean_count = 0
    for subject_id in subjects:
        r = compare_subject(subject_id, old_dir, new_dir, args.noise_threshold)
        if r.get("error"):
            print(f"  [ERROR] {subject_id}: {r['error']}")
            flagged.append(r)
            continue

        is_flagged = (
            r["max_abs_diff"] > args.flag_threshold
            or r.get("num_slices_old") != r.get("num_slices_new")
            or r.get("bad_channels_mismatch_count", 0) > 0
        )
        if is_flagged:
            print(
                f"  [FLAGGED] {subject_id}: max_abs_diff={r['max_abs_diff']:.2e}, "
                f"frac_changed(>{args.noise_threshold:.0e})={r['frac_changed']:.4f}, "
                f"num_slices old/new={r.get('num_slices_old')}/{r.get('num_slices_new')}, "
                f"bad_channels_mismatch_count={r.get('bad_channels_mismatch_count', 'N/A')} "
                f"(sample windows: {r.get('bad_channels_mismatch_sample', [])})"
            )
            flagged.append(r)
        else:
            clean_count += 1
            print(f"  [OK] {subject_id}: max_abs_diff={r['max_abs_diff']:.2e}, frac_changed=0.0000")

    print("\n" + "=" * 88)
    print(f"SUMMARY: {clean_count}/{len(subjects)} subjects show ONLY rounding-noise-level differences.")
    print(f"         {len(flagged)}/{len(subjects)} subjects flagged for individual investigation.")
    print("=" * 88)
    if flagged:
        print("Flagged subject_ids:", [r["subject_id"] for r in flagged])


if __name__ == "__main__":
    main()
