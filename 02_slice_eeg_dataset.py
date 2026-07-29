import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union
import mne
import numpy as np

try:
    import yasa
    HAS_YASA = True
except ImportError:
    HAS_YASA = False

from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".fif", ".edf", ".bdf", ".vhdr"}


def load_raw_eeg(file_path: Path) -> mne.io.BaseRaw:
    """Loads raw EEG recording using MNE."""
    ext = file_path.suffix.lower()
    if ext == ".fif":
        return mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    elif ext == ".edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif ext == ".bdf":
        return mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
    elif ext == ".vhdr":
        return mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def slice_strategy_a_macro(
    raw: mne.io.BaseRaw, window_sec: float = 30.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Strategy A: Standard 30-second contiguous epoch slicing.
    Returns:
        tensor: [Num_Windows, Channels, Window_Samples] (e.g. [N, 64, 6000])
        metadata: List of window timestamps and indices.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # Shape: [Channels, Time_Samples]
    ch_names = raw.ch_names
    
    samples_per_window = int(round(window_sec * sfreq))
    total_samples = data.shape[1]
    num_windows = total_samples // samples_per_window

    slices = []
    metadata = []

    for idx in range(num_windows):
        start_sample = idx * samples_per_window
        end_sample = start_sample + samples_per_window
        
        window_data = data[:, start_sample:end_sample]
        slices.append(window_data)
        
        metadata.append({
            "window_idx": idx,
            "start_sec": start_sample / sfreq,
            "end_sec": end_sample / sfreq,
            "samples": samples_per_window,
            "type": "macro_30s"
        })

    if len(slices) == 0:
        return np.empty((0, len(ch_names), samples_per_window)), []

    tensor = np.stack(slices, axis=0)  # Shape: [N, C, T]
    return tensor, metadata


def slice_strategy_b_micro(
    raw: mne.io.BaseRaw, 
    crop_duration_sec: float = 3.0, 
    freq_band: Tuple[float, float] = (11.0, 16.0),
    threshold_std: float = 1.5
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Strategy B: Event-centered 2-4 second candidate spindle crops.
    Uses YASA if installed; falls back to 11-16 Hz bandpass envelope thresholding.
    Returns:
        tensor: [Num_Events, Channels, Crop_Samples]
        metadata: List of event timestamps and channels.
    """
    sfreq = raw.info["sfreq"]
    crop_samples = int(round(crop_duration_sec * sfreq))
    half_crop = crop_samples // 2
    ch_names = raw.ch_names

    slices = []
    metadata = []

    if HAS_YASA:
        # Use YASA algorithm for spindle detection across central/parietal channels
        try:
            sp = yasa.spindles_detect(raw, verbose=False)
            if sp is not None:
                summary = sp.summary()
                for _, row in summary.iterrows():
                    peak_sec = row["Peak"]
                    peak_sample = int(round(peak_sec * sfreq))
                    
                    start_s = peak_sample - half_crop
                    end_s = peak_sample + half_crop
                    
                    # Ensure boundaries are within array length
                    if start_s >= 0 and end_s <= raw.n_samples:
                        crop_data = raw.get_data(start=start_s, stop=end_s)
                        if crop_data.shape[1] == crop_samples:
                            slices.append(crop_data)
                            metadata.append({
                                "event_channel": row["Channel"],
                                "peak_sec": peak_sec,
                                "duration_sec": row["Duration"],
                                "frequency": row["Frequency"],
                                "type": "event_crop_yasa"
                            })
        except Exception:
            pass  # Fallback to thresholding if YASA detection yields empty or fails

    # Fallback to Envelope Bandpass Peak Detection if YASA not used or no spindles found
    if len(slices) == 0:
        # Filter raw in spindle band (11-16 Hz)
        raw_filtered = raw.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1], verbose=False)
        data_filt = raw_filtered.get_data()
        
        # Calculate envelope Hilbert amplitude across channels
        analytic_signal = mne.filter.filter_data(data_filt, sfreq, freq_band[0], freq_band[1], verbose=False)
        envelope = np.abs(analytic_signal)
        mean_env = np.mean(envelope)
        std_env = np.std(envelope)
        threshold = mean_env + (threshold_std * std_env)

        # Detect candidate peak samples above threshold
        peak_indices = np.where(np.max(envelope, axis=0) > threshold)[0]
        
        # Non-maximum suppression / refractory period (min 1.5 seconds between candidates)
        refractory_samples = int(1.5 * sfreq)
        selected_peaks = []
        last_p = -refractory_samples
        for p in peak_indices:
            if p - last_p >= refractory_samples:
                if p - half_crop >= 0 and p + half_crop <= raw.n_samples:
                    selected_peaks.append(p)
                    last_p = p

        data_raw = raw.get_data()
        for idx, peak in enumerate(selected_peaks):
            start_s = peak - half_crop
            end_s = peak + half_crop
            crop_data = data_raw[:, start_s:end_s]
            if crop_data.shape[1] == crop_samples:
                slices.append(crop_data)
                metadata.append({
                    "event_idx": idx,
                    "peak_sec": peak / sfreq,
                    "duration_sec": crop_duration_sec,
                    "type": "event_crop_bandpass"
                })

    if len(slices) == 0:
        return np.empty((0, len(ch_names), crop_samples)), []

    tensor = np.stack(slices, axis=0)
    return tensor, metadata


def process_subject_slicing_worker(
    args_tuple: Tuple[Path, Path, re.Pattern, str, float, bool]
) -> Dict[str, Union[str, int]]:
    """Worker task executing slicing per subject file."""
    src_file, dst_dir, pattern, strategy, window_sec, force = args_tuple
    
    subject_id = src_file.stem
    if pattern:
        match = pattern.search(subject_id)
        if match:
            subject_id = match.group(0)
    output_npy = dst_dir / f"{subject_id}_windows.npy"
    output_meta = dst_dir / f"{subject_id}_meta.json"

    dst_dir.mkdir(parents=True, exist_ok=True)

    if output_npy.exists() and output_meta.exists() and not force:
        return {"status": "SKIPPED", "subject": subject_id, "count": 0}

    try:
        raw = load_raw_eeg(src_file)

        if strategy.lower() == "macro":
            tensor, meta = slice_strategy_a_macro(raw, window_sec=window_sec)
        elif strategy.lower() == "micro":
            tensor, meta = slice_strategy_b_micro(raw, crop_duration_sec=window_sec)
        else:
            raise ValueError(f"Unknown slicing strategy: {strategy}")

        # Save binary matrix and JSON metadata
        np.save(output_npy, tensor)
        with open(output_meta, "w") as f:
            json.dump({
                "subject_id": subject_id,
                "strategy": strategy,
                "num_channels": len(raw.ch_names),
                "channel_names": raw.ch_names,
                "sampling_freq": raw.info["sfreq"],
                "num_slices": len(meta),
                "slices": meta
            }, f, indent=2)

        return {"status": "SUCCESS", "subject": subject_id, "count": len(meta)}

    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "subject": subject_id, "count": 0}


def run_slicing_pipeline(
    src_dir: Path,
    dst_dir: Path,
    pattern: re.Pattern = None,
    strategy: str = "macro",
    window_sec: float = 30.0,
    num_workers: int = 1,
    force: bool = False
):
    """Executes parallel subject-level window slicing."""
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))
    files = sorted(files)

    print(f"Found {len(files)} files for slicing. Strategy: '{strategy.upper()}', Window: {window_sec}s")
    
    tasks = [
        (f, dst_dir / f.relative_to(src_dir).parent, pattern, strategy, window_sec, force) 
        for f in files
    ]

    results = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0, "TOTAL_WINDOWS": 0}

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_subject_slicing_worker, task) for task in tasks]
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Slicing Dataset", unit="subj")
            for future in pbar:
                res = future.result()
                status = res["status"].split(":")[0]
                results[status] = results.get(status, 0) + 1
                results["TOTAL_WINDOWS"] += res["count"]
                pbar.set_postfix({"Status": status, "Slices": res["count"]})
    else:
        pbar = tqdm(tasks, desc="Slicing Dataset (Single Process)", unit="subj")
        for task in pbar:
            res = process_subject_slicing_worker(task)
            status = res["status"].split(":")[0]
            results[status] = results.get(status, 0) + 1
            results["TOTAL_WINDOWS"] += res["count"]
            pbar.set_postfix({"Status": status, "Slices": res["count"]})

    print("\n=== Slicing Summary ===")
    print(f" - Processed Subjects: {results['SUCCESS']}")
    print(f" - Skipped Subjects:   {results['SKIPPED']}")
    print(f" - Errors:             {results['ERROR']}")
    print(f" - Total Extracted Slices: {results['TOTAL_WINDOWS']}")


def valid_regex(pattern_string):
    # This only runs if the user actually provides the argument
    try:
        return re.compile(pattern_string)
    except re.error:
        raise argparse.ArgumentTypeError(f"Invalid regex: '{pattern_string}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Window Slicing Script")
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing 200Hz resampled files")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output destination directory")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject id")
    parser.add_argument("--strategy", type=str, choices=["macro", "micro"], default="macro", help="Slicing strategy: 'macro' (30s) or 'micro' (2-4s event crops)")
    parser.add_argument("--window_sec", type=float, default=30.0, help="Window duration in seconds (30.0 for macro, 3.0 for micro)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="CPU worker count")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")

    args = parser.parse_args()

    run_slicing_pipeline(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        pattern=args.pattern,
        strategy=args.strategy,
        window_sec=args.window_sec,
        num_workers=args.num_workers,
        force=args.force
    )
