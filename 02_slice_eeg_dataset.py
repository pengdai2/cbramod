import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union
import mne
import yasa
import numpy as np

try:
    import yasa
    HAS_YASA = True
except ImportError:
    HAS_YASA = False

from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".fif", ".edf", ".bdf", ".vhdr"}

# Standard sleep stage string normalization map
STAGE_NORM_MAP = {
    "sleep stage w": "W", "stage w": "W", "wake": "W", "0": "W",
    "sleep stage n1": "N1", "stage 1": "N1", "n1": "N1", "1": "N1",
    "sleep stage n2": "N2", "stage 2": "N2", "n2": "N2", "2": "N2",
    "sleep stage n3": "N3", "stage 3": "N3", "stage 4": "N3", "n3": "N3", "3": "N3", "4": "N3",
    "sleep stage r": "REM", "stage rem": "REM", "rem": "REM", "5": "REM"
}


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


def extract_epoch_stages(raw: mne.io.BaseRaw, num_windows: int, window_sec: float = 30.0) -> List[str]:
    """
    Parses MNE raw annotations and maps each window to its corresponding sleep stage.
    """
    if len(raw.annotations) == 0:
        return ["UNKNOWN"] * num_windows

    stages = []
    annotations = raw.annotations

    for idx in range(num_windows):
        t_mid = (idx * window_sec) + (window_sec / 2.0)
        stage_label = "UNKNOWN"

        # Find annotation overlapping the midpoint of this window
        for ann in annotations:
            ann_start = ann["onset"]
            ann_end = ann_start + ann["duration"]
            if ann_start <= t_mid < ann_end:
                raw_desc = str(ann["description"]).strip().lower()
                stage_label = STAGE_NORM_MAP.get(raw_desc, ann["description"])
                break

        stages.append(stage_label)

    return stages


def select_best_eeg_channel(eeg_chs: List[str]) -> str:
    """
    Selects the optimal central EEG channel using exact word-boundary matching.
    Avoids false positive substring matches like FC4 or CP4.
    """
    # Priority list of central channels
    preferences = ["C4", "C3", "CZ"]

    for pref in preferences:
        # \b ensures exact word boundary match (e.g. matches "C4", "C4-M1", "C4_A1", but NOT "FC4")
        pattern = re.compile(rf"\b{pref}\b", re.IGNORECASE)
        for ch in eeg_chs:
            if pattern.search(ch):
                return ch

    # Fallback: Return first available channel if no central channels match
    return eeg_chs[0]


def predict_epoch_stages_yasa(raw: mne.io.BaseRaw, num_windows: int) -> List[str]:
    """Uses YASA to predict sleep stages for unannotated recordings."""
    try:
        eeg_chs = raw.copy().pick_types(eeg=True).ch_names
        if not eeg_chs:
            eeg_chs = raw.ch_names
        target_ch = select_best_eeg_channel(eeg_chs)

        sls = yasa.SleepStaging(raw, eeg_name=target_ch)
        stages = list(sls.predict().hypno)

        # Align predicted length with target window count
        if len(stages) < num_windows:
            stages.extend(["UNKNOWN"] * (num_windows - len(stages)))
        elif len(stages) > num_windows:
            stages = stages[:num_windows]

        return stages
    except Exception as e:
        return ["UNKNOWN"] * num_windows


def slice_strategy_a_macro(
    raw: mne.io.BaseRaw, window_sec: float = 30.0
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Strategy A: Standard 30-second contiguous epoch slicing with stage extraction.
    Returns:
        tensor: [Num_Windows, Channels, Window_Samples]
        metadata: List of window timestamps and indices.
        stages: List of sleep stage labels matching window count.
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
        return np.empty((0, len(ch_names), samples_per_window)), [], []

    tensor = np.stack(slices, axis=0)  # Shape: [N, C, T]
    
    # Check if raw has native annotations; if not, use YASA predictor
    if len(raw.annotations) > 0:
        # Native annotation extraction
        stages = extract_epoch_stages(raw, num_windows=num_windows, window_sec=window_sec)
    else:
        # Automated prediction via YASA
        stages = predict_epoch_stages_yasa(raw, num_windows=num_windows)

    return tensor, metadata, stages


def slice_strategy_b_micro(
    raw: mne.io.BaseRaw, 
    crop_duration_sec: float = 3.0, 
    freq_band: Tuple[float, float] = (11.0, 16.0),
    threshold_std: float = 1.5
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Strategy B: Event-centered candidate spindle crops.
    """
    sfreq = raw.info["sfreq"]
    crop_samples = int(round(crop_duration_sec * sfreq))
    half_crop = crop_samples // 2
    ch_names = raw.ch_names

    slices = []
    metadata = []

    if HAS_YASA:
        try:
            sp = yasa.spindles_detect(raw, verbose=False)
            if sp is not None:
                summary = sp.summary()
                for _, row in summary.iterrows():
                    peak_sec = row["Peak"]
                    peak_sample = int(round(peak_sec * sfreq))
                    
                    start_s = peak_sample - half_crop
                    end_s = peak_sample + half_crop
                    
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
            pass

    if len(slices) == 0:
        raw_filtered = raw.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1], verbose=False)
        data_filt = raw_filtered.get_data()
        
        analytic_signal = mne.filter.filter_data(data_filt, sfreq, freq_band[0], freq_band[1], verbose=False)
        envelope = np.abs(analytic_signal)
        mean_env = np.mean(envelope)
        std_env = np.std(envelope)
        threshold = mean_env + (threshold_std * std_env)

        peak_indices = np.where(np.max(envelope, axis=0) > threshold)[0]
        
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
        return np.empty((0, len(ch_names), crop_samples)), [], []

    tensor = np.stack(slices, axis=0)

    # Check if raw has native annotations; if not, use YASA predictor
    if len(raw.annotations) > 0:
        # Native annotation extraction
        stages = extract_epoch_stages(raw, num_windows=num_windows, window_sec=window_sec)
    else:
        # Automated prediction via YASA
        stages = predict_epoch_stages_yasa(raw, num_windows=num_windows)

    return tensor, metadata, stages


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
            tensor, meta, stages = slice_strategy_a_macro(raw, window_sec=window_sec)
        elif strategy.lower() == "micro":
            tensor, meta, stages = slice_strategy_b_micro(raw, crop_duration_sec=window_sec)
        else:
            raise ValueError(f"Unknown slicing strategy: {strategy}")

        # Save binary matrix and JSON metadata WITH top-level stages key
        np.save(output_npy, tensor)
        with open(output_meta, "w") as f:
            json.dump({
                "subject_id": subject_id,
                "strategy": strategy,
                "num_channels": len(raw.ch_names),
                "channel_names": raw.ch_names,
                "sampling_freq": raw.info["sfreq"],
                "num_slices": len(meta),
                "stages": stages,  # Top-level array consumed by 09_clinical_inference.py
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
    try:
        return re.compile(pattern_string)
    except re.error:
        raise argparse.ArgumentTypeError(f"Invalid regex: '{pattern_string}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Window Slicing Script")
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing resampled files")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output destination directory")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject id")
    parser.add_argument("--strategy", type=str, choices=["macro", "micro"], default="macro", help="Slicing strategy: 'macro' (30s) or 'micro'")
    parser.add_argument("--window_sec", type=float, default=30.0, help="Window duration in seconds")
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
