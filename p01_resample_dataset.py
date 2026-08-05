import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
from cbramod_utils import find_eeg_files, load_raw_eeg
import mne
import numpy as np
from tqdm import tqdm

def compute_resample_ratio(orig_sfreq: float, target_sfreq: float) -> Tuple[int, int]:
    """Computes exact integer up/down factors for non-divisive resampling."""
    orig_int = int(round(orig_sfreq))
    target_int = int(round(target_sfreq))
    common_gcd = math.gcd(orig_int, target_int)
    up = target_int // common_gcd
    down = orig_int // common_gcd
    return up, down

def validate_resampled_data(
    raw_resampled: mne.io.BaseRaw, 
    expected_sfreq: float, 
    expected_channels: int, 
    expected_duration_sec: float
) -> bool:
    """Validates output data integrity."""
    if not np.isclose(raw_resampled.info["sfreq"], expected_sfreq, atol=1e-2):
        return False
    if len(raw_resampled.ch_names) != expected_channels:
        return False
    if not np.isclose(raw_resampled.times[-1], expected_duration_sec, atol=1.0 / expected_sfreq):
        return False
    
    data = raw_resampled.get_data()
    if np.isnan(data).any() or np.isinf(data).any():
        return False
    
    return True

def process_file_worker(args_tuple: Tuple[Path, Path, float, bool]) -> Dict[str, str]:
    """
    Top-level worker function executed by each worker process.
    Unpacks tuple arguments for pickling compatibility across processes.
    """
    src_file, dst_file, target_sfreq, force_reprocess = args_tuple
    
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Resumability check
    if dst_file.exists() and not force_reprocess:
        try:
            resample_check = load_raw_eeg(dst_file)
            if np.isclose(resample_check.info["sfreq"], target_sfreq, atol=1e-2):
                return {"status": "SKIPPED", "file": str(src_file.name)}
        except Exception:
            pass  # Re-process if file is corrupted

    try:
        raw = load_raw_eeg(src_file)
        orig_sfreq = raw.info["sfreq"]
        orig_channels = len(raw.ch_names)
        orig_duration = raw.times[-1]

        # Copy over directly if already at target sampling frequency
        if np.isclose(orig_sfreq, target_sfreq, atol=1e-2):
            raw.save(dst_file, overwrite=True, verbose=False)
            return {"status": "COPIED", "file": str(src_file.name)}

        # Perform non-divisive resampling
        raw_resampled = raw.copy().resample(sfreq=target_sfreq, npad="auto", verbose=False)

        # Integrity Validation
        valid = validate_resampled_data(
            raw_resampled, 
            expected_sfreq=target_sfreq, 
            expected_channels=orig_channels, 
            expected_duration_sec=orig_duration
        )
        
        if not valid:
            return {"status": "FAILED_VALIDATION", "file": str(src_file.name)}

        # Export resampled signal
        raw_resampled.save(dst_file, overwrite=True, verbose=False)

        return {"status": "SUCCESS", "file": str(src_file.name)}

    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "file": str(src_file.name)}

def run_resampling_pipeline(
    src_dir: Path, 
    dst_dir: Path, 
    target_sfreq: float, 
    num_workers: int,
    force_reprocess: bool = False
):
    """Executes multi-threaded/multi-process cohort resampling with tqdm tracking."""
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()
    
    eeg_files = find_eeg_files(src_dir)
    print(f"Found {len(eeg_files)} EEG files across subject directories.")
    print(f"Executing parallel processing across {num_workers} CPU workers...")
    
    # Build argument tuples for multiprocessing map
    tasks = []
    for src_file in eeg_files:
        relative_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative_path
        if dst_file.suffix.lower() != ".fif":
            dst_file = dst_file.with_suffix(".fif")
        tasks.append((src_file, dst_file, target_sfreq, force_reprocess))

    results = {"SUCCESS": 0, "SKIPPED": 0, "COPIED": 0, "FAILED_VALIDATION": 0, "ERROR": 0}

    # Execute workers in parallel using ProcessPoolExecutor
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file_worker, task) for task in tasks]
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Resampling Cohort", unit="file")
            
            for future in pbar:
                res = future.result()
                status_key = res["status"].split(":")[0]
                results[status_key] = results.get(status_key, 0) + 1
                pbar.set_postfix({"Last Status": res["status"], "File": res["file"][:15]})
    else:
        # Single-process fallback mode (for debugging)
        pbar = tqdm(tasks, desc="Resampling Cohort (Single Process)", unit="file")
        for task in pbar:
            res = process_file_worker(task)
            status_key = res["status"].split(":")[0]
            results[status_key] = results.get(status_key, 0) + 1
            pbar.set_postfix({"Last Status": res["status"], "File": res["file"][:15]})

    print("\n=== Resampling Summary ===")
    for status, count in results.items():
        print(f" - {status}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel EEG Resampling Pipeline for CBraMod")
    parser.add_argument("--src_dir", type=str, required=True, help="Path to raw top-level cohort directory")
    parser.add_argument("--dst_dir", type=str, required=True, help="Path to parallel destination directory")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate (Hz) [Default: 200.0]")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help=f"Number of CPU workers [Default: {os.cpu_count()}]")
    parser.add_argument("--force", action="store_true", help="Force reprocessing existing files")
    
    args = parser.parse_args()
    
    run_resampling_pipeline(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        target_sfreq=args.target_sfreq,
        num_workers=args.num_workers,
        force_reprocess=args.force
    )
