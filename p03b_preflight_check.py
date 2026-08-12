"""
Preflight Data Sanity Check CLI for Sleep EEG Pipeline.

Performs rigorous pre-training validation on manifest CSVs, .npy tensor files, 
and metadata JSONs. Handles zero-padded channels gracefully while detecting 
missing files, corrupted arrays, NaN/Inf values, shape mismatches, true 
flatlined channels, and label anomalies before fine-tuning.

Usage:
    python preflight_check.py --manifest-dir ./manifests --data-dir ./data --expected-channels 64
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class ConsoleColor:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_pass(msg: str):
    print(f"[{ConsoleColor.GREEN}PASS{ConsoleColor.RESET}] {msg}")


def log_warn(msg: str):
    print(f"[{ConsoleColor.YELLOW}WARN{ConsoleColor.RESET}] {msg}")


def log_fail(msg: str):
    print(f"[{ConsoleColor.RED}FAIL{ConsoleColor.RESET}] {msg}")


def log_info(msg: str):
    print(f"[{ConsoleColor.CYAN}INFO{ConsoleColor.RESET}] {msg}")


class EEGPreflightChecker:
    def __init__(
        self,
        manifest_dir: Path,
        data_dir: Optional[Path] = None,
        expected_channels: int = 64,
        sample_check_ratio: float = 0.2,
        flatline_std_threshold: float = 1e-7,
        extreme_value_threshold: float = 1e4,
        allow_zero_padding: bool = True,
        min_active_channels: int = 1
    ):
        self.manifest_dir = Path(manifest_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.expected_channels = expected_channels
        self.sample_check_ratio = max(0.01, min(1.0, sample_check_ratio))
        self.flatline_std_threshold = flatline_std_threshold
        self.extreme_value_threshold = extreme_value_threshold
        self.allow_zero_padding = allow_zero_padding
        self.min_active_channels = min_active_channels

        self.errors_found = 0
        self.warnings_found = 0

    def resolve_path(self, raw_path: Union[str, Path]) -> Path:
        p = Path(raw_path)
        if p.is_absolute() or self.data_dir is None:
            return p
        return self.data_dir / p

    def validate_manifest_file(self, manifest_path: Path) -> Optional[pd.DataFrame]:
        log_info(f"Validating manifest file structure: {manifest_path}")
        if not manifest_path.exists():
            log_fail(f"Manifest file does not exist: {manifest_path}")
            self.errors_found += 1
            return None

        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            log_fail(f"Failed to parse CSV manifest {manifest_path}: {e}")
            self.errors_found += 1
            return None

        required_cols = {"subject_id", "npy_path", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            log_fail(f"Manifest missing required columns: {missing_cols}")
            self.errors_found += 1
            return None

        log_pass(f"Manifest loaded successfully ({len(df)} subject entries found).")
        return df

    def inspect_subject_data(
        self,
        row: pd.Series,
        deep_inspect: bool = False
    ) -> Dict[str, Union[int, float, bool, List[str]]]:
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "num_slices": 0,
            "valid_slices": 0,
            "invalid_slices": 0,
            "nan_count": 0,
            "inf_count": 0,
            "flatlines_detected": 0,
            "padded_channels": 0,
            "active_channels": self.expected_channels
        }

        subject_id = row["subject_id"]
        npy_path = self.resolve_path(row["npy_path"])
        meta_path = self.resolve_path(row["meta_path"]) if "meta_path" in row and pd.notna(row["meta_path"]) else None
        label = row["label"]

        # Check Label Integrity
        if pd.isna(label) or int(label) == -1:
            results["warnings"].append(f"Subject {subject_id}: Unlabeled or assigned -1 label.")

        # Check .npy existence
        if not npy_path.exists():
            results["valid"] = False
            results["errors"].append(f"Subject {subject_id}: Tensor file missing at {npy_path}")
            return results

        # Memory-map tensor inspection
        try:
            mmap_arr = np.load(npy_path, mmap_mode="r")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Subject {subject_id}: Failed to load tensor {npy_path} ({e})")
            return results

        # Shape validation
        shape = mmap_arr.shape
        if len(shape) != 3:
            results["valid"] = False
            results["errors"].append(
                f"Subject {subject_id}: Expected 3D array [Windows, Channels, Time], got shape {shape}"
            )
            return results

        num_windows, num_channels, time_samples = shape
        results["num_slices"] = num_windows

        if num_channels != self.expected_channels:
            results["errors"].append(
                f"Subject {subject_id}: Channel mismatch. Expected {self.expected_channels}, got {num_channels}"
            )
            results["valid"] = False

        if time_samples < 100:
            results["warnings"].append(
                f"Subject {subject_id}: Exceptionally short time length ({time_samples} samples per window)."
            )

        # Meta JSON Validation
        valid_indices = list(range(num_windows))
        if meta_path:
            if not meta_path.exists():
                results["warnings"].append(f"Subject {subject_id}: Metadata file specified but missing: {meta_path}")
            else:
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)

                    slices_meta = meta.get("slices", [])
                    if slices_meta and len(slices_meta) != num_windows:
                        results["warnings"].append(
                            f"Subject {subject_id}: Metadata slice count ({len(slices_meta)}) "
                            f"differs from tensor shape count ({num_windows})."
                        )

                    valid_indices = [
                        s["window_idx"] for s in slices_meta 
                        if s.get("is_valid", True) and s.get("window_idx", 0) < num_windows
                    ] if slices_meta else list(range(num_windows))

                    results["valid_slices"] = len(valid_indices)
                    results["invalid_slices"] = num_windows - len(valid_indices)

                except Exception as e:
                    results["warnings"].append(f"Subject {subject_id}: Malformed JSON metadata {meta_path}: {e}")
        else:
            results["valid_slices"] = num_windows

        # Deep Numerical Array Inspection (Sampled or Full)
        if deep_inspect and results["valid"] and len(valid_indices) > 0:
            try:
                # Load valid slices into memory for checking
                sample_slice_indices = valid_indices[:min(len(valid_indices), 20)]
                arr_chunk = np.array(mmap_arr[sample_slice_indices], dtype=np.float32)  # [Slices, Channels, Time]

                # Check NaN / Inf
                nans = np.isnan(arr_chunk).sum()
                infs = np.isinf(arr_chunk).sum()

                if nans > 0:
                    results["errors"].append(f"Subject {subject_id}: Detected {nans} NaN values in sampled tensors.")
                    results["valid"] = False
                    results["nan_count"] += nans

                if infs > 0:
                    results["errors"].append(f"Subject {subject_id}: Detected {infs} Inf values in sampled tensors.")
                    results["valid"] = False
                    results["inf_count"] += infs

                # Zero-Padded Channels vs. Active Channels Analysis
                # A channel is zero-padded if all time samples across all inspected windows are identically zero
                is_channel_zero_padded = np.all(arr_chunk == 0, axis=(0, 2))  # Shape: [Channels]
                num_padded = int(np.sum(is_channel_zero_padded))
                num_active = self.expected_channels - num_padded

                results["padded_channels"] = num_padded
                results["active_channels"] = num_active

                if not self.allow_zero_padding and num_padded > 0:
                    results["errors"].append(
                        f"Subject {subject_id}: Zero-padded channels detected ({num_padded} padded), but --allow_zero_padding is False."
                    )
                    results["valid"] = False

                if num_active < self.min_active_channels:
                    results["errors"].append(
                        f"Subject {subject_id}: Insufficient active channels ({num_active} active < min required {self.min_active_channels})."
                    )
                    results["valid"] = False

                # Check for extreme scale outliers on ACTIVE channels only
                if num_active > 0:
                    active_chunk = arr_chunk[:, ~is_channel_zero_padded, :]
                    max_val = np.abs(active_chunk).max()
                    if max_val > self.extreme_value_threshold:
                        results["warnings"].append(
                            f"Subject {subject_id}: Extreme unscaled values detected on active channels (Max abs: {max_val:.2f})."
                        )

                    # Check standard deviation across time to catch flatlines ONLY on active channels
                    stds = np.std(active_chunk, axis=-1)  # Shape: [Slices, Active_Channels]
                    flatlines = np.sum(stds < self.flatline_std_threshold)
                    if flatlines > 0:
                        results["warnings"].append(
                            f"Subject {subject_id}: Detected {flatlines} flatlined/zero-variance active channel windows."
                        )
                        results["flatlines_detected"] += flatlines

            except Exception as e:
                results["warnings"].append(f"Subject {subject_id}: Deep numerical check failed: {e}")

        return results

    def run_check(self, manifest_file_name: str) -> bool:
        manifest_path = self.manifest_dir / manifest_file_name
        print("\n" + "=" * 75)
        log_info(f"STARTING PREFLIGHT SANITY CHECK: {manifest_file_name}")
        print("=" * 75)

        df = self.validate_manifest_file(manifest_path)
        if df is None:
            return False

        total_subjects = len(df)
        total_slices = 0
        total_valid_slices = 0
        total_padded_channels_list = []
        label_counter: Dict[int, int] = {}

        # Determine indices for deep numerical array testing
        deep_check_count = max(1, int(total_subjects * self.sample_check_ratio))
        deep_check_indices = set(np.random.choice(total_subjects, size=deep_check_count, replace=False))

        log_info(
            f"Checking {total_subjects} subjects "
            f"(performing deep tensor inspection on {deep_check_count} random samples)..."
        )

        start_time = time.time()
        for idx, row in tqdm(df.iterrows(), total=total_subjects, desc="Auditing Subjects"):
            label = int(row["label"]) if pd.notna(row["label"]) else -1
            label_counter[label] = label_counter.get(label, 0) + 1

            is_deep = idx in deep_check_indices
            subject_res = self.inspect_subject_data(row, deep_inspect=is_deep)

            total_slices += subject_res["num_slices"]
            total_valid_slices += subject_res["valid_slices"]

            if is_deep and subject_res["valid"]:
                total_padded_channels_list.append(subject_res["padded_channels"])

            for err in subject_res["errors"]:
                log_fail(err)
                self.errors_found += 1

            for warn in subject_res["warnings"]:
                log_warn(warn)
                self.warnings_found += 1

        elapsed = time.time() - start_time
        avg_padded = np.mean(total_padded_channels_list) if total_padded_channels_list else 0.0

        # Print Split Summary
        print("\n" + "-" * 45)
        print(f"{ConsoleColor.BOLD}SUMMARY REPORT: {manifest_file_name}{ConsoleColor.RESET}")
        print("-" * 45)
        print(f"  Processed Subjects:        {total_subjects:,}")
        print(f"  Total Extracted Slices:    {total_slices:,}")
        print(f"  Usable (Valid) Slices:     {total_valid_slices:,} "
              f"({(total_valid_slices/max(1, total_slices))*100:.1f}%)")
        print(f"  Avg Active Channels:       {self.expected_channels - avg_padded:.1f} / {self.expected_channels}")
        print(f"  Avg Zero-Padded Channels:  {avg_padded:.1f} / {self.expected_channels}")
        print(f"  Label Distribution:        {dict(sorted(label_counter.items()))}")
        print(f"  Audit Duration:            {elapsed:.2f}s")
        print("-" * 45)

        return self.errors_found == 0


def main():
    parser = argparse.ArgumentParser(
        description="Preflight Data Sanity Check for Sleep EEG Fine-Tuning Pipeline."
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        required=True,
        help="Directory containing train_manifest.csv and val_manifest.csv"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Top-level root directory where relative tensor/meta paths reside"
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=64,
        help="Expected total tensor channel count (default: 64)"
    )
    parser.add_argument(
        "--min-active-channels",
        type=int,
        default=1,
        help="Minimum non-zero active channels required per subject (default: 1)"
    )
    parser.add_argument(
        "--disallow-zero-padding",
        action="store_true",
        help="Raise error if any zero-padded channel is detected"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.2,
        help="Fraction of files (0.0 - 1.0) to perform deep numerical array inspection on (default: 0.2)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit immediately with non-zero error code if warnings or errors are found"
    )

    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None

    checker = EEGPreflightChecker(
        manifest_dir=manifest_dir,
        data_dir=data_dir,
        expected_channels=args.expected_channels,
        sample_check_ratio=args.sample_ratio,
        allow_zero_padding=not args.disallow_zero_padding,
        min_active_channels=args.min_active_channels
    )

    train_ok = checker.run_check("train_manifest.csv")
    val_ok = checker.run_check("val_manifest.csv")

    print("\n" + "=" * 75)
    if checker.errors_found == 0 and checker.warnings_found == 0:
        log_pass(f"{ConsoleColor.BOLD}PREFLIGHT CHECK PASSED: Dataset is clean and ready for training!{ConsoleColor.RESET}")
        sys.exit(0)
    elif checker.errors_found == 0:
        log_warn(
            f"{ConsoleColor.BOLD}PREFLIGHT COMPLETED WITH {checker.warnings_found} WARNING(S). "
            f"Review messages above before training.{ConsoleColor.RESET}"
        )
        sys.exit(1 if args.strict else 0)
    else:
        log_fail(
            f"{ConsoleColor.BOLD}PREFLIGHT CHECK FAILED: Found {checker.errors_found} Critical Error(s) "
            f"and {checker.warnings_found} Warning(s).{ConsoleColor.RESET}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
