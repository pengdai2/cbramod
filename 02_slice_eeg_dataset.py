#!/usr/bin/env python3
"""
EEG Window Slicing Pipeline with Top-Level Continuous Cleansing & Quality Tracking.

Architecture Pipeline:
1. Load Raw File -> Read continuous time-series recording (.edf, .fif, .bdf, .vhdr).
2. Dynamic Discovery -> Identify recorded scalp channels matching CBraMod layout at runtime.
3. Channel Selection -> Crop away non-EEG/trigger channels to protect square pulses from FIR ringing.
4. Continuous Filtering -> Apply single-pass 0.3-35.0 Hz zero-phase FIR bandpass filter globally 
   to eliminate baseline drift and sweat sway BEFORE calculating standard deviations.
5. Referencing -> Apply linked-earlobe (A1/A2) or CAR referencing.
6. Neighborhood Spatial Validation & Interpolation -> Detect flatlines/high-variance noise, check 
   spatial adjacency (reject spatial clusters of bad channels), and interpolate isolated bad channels via spherical splines.
7. CBraMod Harmonization -> Map cleaned physical signals into 64-channel matrix; zero-pad unrecorded positions (0.0 uV).
8. Slicing & Normalization -> Epoch time-series into windows and apply per-window Z-score normalization 
   strictly to active channels.
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union
import mne
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm

try:
    import yasa
    HAS_YASA = True
except ImportError:
    HAS_YASA = False

SUPPORTED_EXTENSIONS = {".fif", ".edf", ".bdf", ".vhdr"}

# Standard CBraMod 64-channel 10-20 spatial layout topology
CBRMOD_STANDARD_64 = [
    'FP1', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'AF3', 'AF7',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
    'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'FC1', 'FC3',
    'FC5', 'FT7', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2',
    'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'CPZ',
    'CP1', 'CP3', 'CP5', 'TP7', 'P7', 'P5', 'P3', 'P1',
    'PZ', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'POZ',
    'PO3', 'PO7', 'O1', 'OZ', 'O2', 'CB1', 'CB2', 'IZ'
]

# Standard sleep stage string normalization map
STAGE_NORM_MAP = {
    "sleep stage w": "W", "stage w": "W", "wake": "W", "0": "W",
    "sleep stage n1": "N1", "stage 1": "N1", "n1": "N1", "1": "N1",
    "sleep stage n2": "N2", "stage 2": "N2", "n2": "N2", "2": "N2",
    "sleep stage n3": "N3", "stage 3": "N3", "stage 4": "N3", "n3": "N3", "3": "N3", "4": "N3",
    "sleep stage r": "REM", "stage rem": "REM", "rem": "REM", "5": "REM"
}


def clean_ch_name(name: str) -> str:
    """Standardizes channel naming for robust string matching."""
    return name.upper().replace('.', '').replace('-', '').replace('EEG', '').strip()


def discover_cbramod_channels(raw: mne.io.BaseRaw) -> List[str]:
    """
    Dynamically discovers recorded scalp channels that correspond 
    to standard CBraMod 10-20 EEG positions at runtime.
    """
    clean_target_map = {clean_ch_name(ch): ch for ch in CBRMOD_STANDARD_64}
    matched_channels = []
    
    for ch in raw.ch_names:
        cleaned = clean_ch_name(ch)
        if cleaned in clean_target_map:
            matched_channels.append(ch)
            
    return matched_channels


def load_raw_eeg(file_path: Path) -> mne.io.BaseRaw:
    """Loads raw continuous EEG recording using MNE."""
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


def apply_eeg_referencing(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Applies linked-earlobe referencing (A1/A2) if present,
    else falls back to Common Average Referencing (CAR).
    """
    ch_clean = [clean_ch_name(ch) for ch in raw.ch_names]
    if "A1" in ch_clean and "A2" in ch_clean:
        a1_ch = raw.ch_names[ch_clean.index("A1")]
        a2_ch = raw.ch_names[ch_clean.index("A2")]
        raw.set_eeg_reference(ref_channels=[a1_ch, a2_ch], projection=False, verbose=False)
    else:
        try:
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
        except Exception:
            pass
    return raw


def validate_spatial_neighborhood(
    raw: mne.io.BaseRaw, 
    bad_ch_names: List[str], 
    max_bad_neighbors: int = 1
) -> bool:
    """
    Evaluates physical adjacency of bad channels using the 10-20 topology matrix.
    Returns False if any bad channel is adjacent to more than `max_bad_neighbors` bad channels,
    preventing spherical spline interpolation across unanchored spatial clusters.
    """
    if len(bad_ch_names) <= 1:
        return True  # Single bad channel is spatially isolated

    try:
        adjacency_matrix, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type='eeg')
        adj_dense = adjacency_matrix.toarray()
        
        ch_map = {name: idx for idx, name in enumerate(ch_names)}
        bad_indices = [ch_map[ch] for ch in bad_ch_names if ch in ch_map]

        for bad_idx in bad_indices:
            neighbor_indices = np.where(adj_dense[bad_idx])[0]
            bad_neighbor_count = sum(1 for idx in neighbor_indices if idx in bad_indices)
            
            if bad_neighbor_count > max_bad_neighbors:
                return False  # Spatial cluster of bad channels detected
        return True
    except Exception:
        # Fallback if adjacency matrix cannot be derived; treat as safe
        return True


def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    recorded_ch_names: List[str],
    min_std_uV: float = 0.5,
    max_std_uV: float = 150.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1
) -> Tuple[mne.io.BaseRaw, List[str], bool]:
    """
    Attaches 3D spatial montage, checks overall bad channel ratio and spatial neighborhood 
    clustering, and interpolates isolated bad channels via spherical splines.
    """
    n_recorded = len(recorded_ch_names)
    if n_recorded == 0:
        return raw, [], False

    # Attach 3D spatial coordinates required for adjacency checks & spherical spline interpolation
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # Evaluate standard deviations on continuous physical channel data
    data_uV = raw.get_data() * 1e6
    ch_stds = np.std(data_uV, axis=-1)

    bad_mask = (ch_stds < min_std_uV) | (ch_stds > max_std_uV)
    bad_indices = np.where(bad_mask)[0]
    bad_ch_names = [raw.ch_names[i] for i in bad_indices]

    # 1. Reject recording if total bad physical channel ratio exceeds limit
    bad_ratio = len(bad_ch_names) / float(n_recorded)
    if bad_ratio > max_bad_ratio:
        return raw, bad_ch_names, False

    # 2. Reject recording if bad channels form spatial clusters (adjacent bad neighbors)
    if len(bad_ch_names) > 1:
        is_spatially_isolated = validate_spatial_neighborhood(
            raw, bad_ch_names, max_bad_neighbors=max_bad_neighbors
        )
        if not is_spatially_isolated:
            return raw, bad_ch_names, False

    # 3. Interpolate isolated bad physical sensors using clean neighboring electrodes
    if len(bad_ch_names) > 0:
        raw.info['bads'] = bad_ch_names
        raw.interpolate_bads(reset=True, mode='accurate', verbose=False)

    return raw, bad_ch_names, True


def prepare_clean_raw_eeg(
    file_path: Path, 
    l_freq: float = 0.3, 
    h_freq: float = 35.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, List[str], List[str], bool]:
    """
    TOP-LEVEL DATA PREPARATION PIPELINE:
    Loads continuous recording, isolates physical EEG channels, applies single-pass 
    zero-phase FIR bandpass filtering, re-references, validates spatial neighborhood, 
    and repairs bad channels prior to slicing.
    
    Returns:
        (raw_orig, raw_clean_eeg, recorded_eeg_chs, bad_chs, subject_ok)
    """
    # 1. Load continuous raw recording
    raw_orig = load_raw_eeg(file_path)

    # 2. Dynamic runtime discovery of physical EEG channels matching CBraMod layout
    recorded_eeg_chs = discover_cbramod_channels(raw_orig)
    if len(recorded_eeg_chs) == 0:
        return raw_orig, raw_orig, [], [], False

    # 3. Pick physical scalp channels (drops non-EEG/STATUS/trigger channels BEFORE filtering)
    raw_eeg = raw_orig.copy().pick_channels(recorded_eeg_chs, ordered=True)

    # 4. Continuous zero-phase FIR bandpass filtering (0.3 Hz - 35.0 Hz)
    #    Single global filtering pass applied to continuous data
    raw_eeg.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )

    # 5. Apply re-referencing (CAR / A1-A2)
    raw_ref = apply_eeg_referencing(raw_eeg)

    # 6. Detect bad channels, validate neighborhood topology, and execute spherical spline interpolation
    raw_clean_eeg, bad_chs, subject_ok = detect_and_interpolate_bad_channels(
        raw_ref, 
        recorded_ch_names=recorded_eeg_chs, 
        max_bad_ratio=max_bad_ratio,
        max_bad_neighbors=max_bad_neighbors
    )

    return raw_orig, raw_clean_eeg, recorded_eeg_chs, bad_chs, subject_ok


def harmonize_channels_to_cbramod(data_uV: np.ndarray, orig_ch_names: List[str]) -> np.ndarray:
    """
    Maps cleaned physical channels into CBraMod's standard 64-channel matrix.
    Unrecorded channels are left zero-padded (0.0 uV) to preserve spatial topology.
    """
    clean_orig = [clean_ch_name(ch) for ch in orig_ch_names]
    clean_target = [clean_ch_name(ch) for ch in CBRMOD_STANDARD_64]

    num_samples = data_uV.shape[1]
    harmonized = np.zeros((len(CBRMOD_STANDARD_64), num_samples), dtype=np.float32)

    for t_idx, t_ch in enumerate(clean_target):
        if t_ch in clean_orig:
            s_idx = clean_orig.index(t_ch)
            harmonized[t_idx, :] = data_uV[s_idx, :]

    return harmonized


def process_and_normalize_slice(
    slice_64: np.ndarray, min_std: float = 0.5, max_std: float = 150.0
) -> Tuple[np.ndarray, bool, str]:
    """
    Screens window active channels for residual flatlines/artifacts.
    Applies per-window Z-score normalization strictly across active (non-zero padded) channels.
    """
    nonzero_mask = np.abs(slice_64).sum(axis=-1) > 1e-8

    if not np.any(nonzero_mask):
        return np.zeros_like(slice_64, dtype=np.float32), False, "EMPTY_CHANNEL_DATA"

    active_stds = slice_64[nonzero_mask].std(axis=-1)

    if np.any(active_stds < min_std):
        return np.zeros_like(slice_64, dtype=np.float32), False, "FLATLINE_DETECTED"

    if np.any(active_stds > max_std):
        return np.zeros_like(slice_64, dtype=np.float32), False, "EXTREME_ARTIFACT"

    # Z-score normalize active channels
    normalized = slice_64.copy()
    means = normalized[nonzero_mask].mean(axis=-1, keepdims=True)
    stds = normalized[nonzero_mask].std(axis=-1, keepdims=True)

    normalized[nonzero_mask] = (normalized[nonzero_mask] - means) / (stds + 1e-8)
    return normalized, True, "OK"


def extract_epoch_stages(raw: mne.io.BaseRaw, num_windows: int, window_sec: float = 30.0) -> List[str]:
    """Parses MNE raw annotations and maps each epoch to its sleep stage label."""
    if len(raw.annotations) == 0:
        return ["UNKNOWN"] * num_windows

    stages = []
    annotations = raw.annotations

    for idx in range(num_windows):
        t_mid = (idx * window_sec) + (window_sec / 2.0)
        stage_label = "UNKNOWN"

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
    """Selects central EEG channel for YASA prediction using regex word boundary matching."""
    preferences = ["C4", "C3", "CZ"]

    for pref in preferences:
        pattern = re.compile(rf"\b{pref}\b", re.IGNORECASE)
        for ch in eeg_chs:
            if pattern.search(ch):
                return ch

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

        if len(stages) < num_windows:
            stages.extend(["UNKNOWN"] * (num_windows - len(stages)))
        elif len(stages) > num_windows:
            stages = stages[:num_windows]

        return stages
    except Exception:
        return ["UNKNOWN"] * num_windows


def slice_strategy_a_macro(
    raw_orig: mne.io.BaseRaw,
    raw_clean_eeg: mne.io.BaseRaw,
    recorded_chs: List[str],
    bad_chs: List[str],
    subject_ok: bool,
    window_sec: float = 30.0
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Strategy A: Contiguous 30-second epoch slicing.
    Unconditionally outputs every 30s block to preserve 1-to-1 temporal indexing alignment.
    """
    sfreq = raw_clean_eeg.info["sfreq"]
    samples_per_window = int(round(window_sec * sfreq))
    total_samples = raw_clean_eeg.n_samples
    num_windows = total_samples // samples_per_window

    if len(raw_orig.annotations) > 0:
        raw_stages = extract_epoch_stages(raw_orig, num_windows=num_windows, window_sec=window_sec)
    else:
        raw_stages = predict_epoch_stages_yasa(raw_clean_eeg, num_windows=num_windows)

    if not subject_ok:
        slices = [np.zeros((64, samples_per_window), dtype=np.float32) for _ in range(num_windows)]
        metadata = [{
            "window_idx": idx,
            "is_valid": False,
            "quality_status": "EXCESSIVE_OR_CLUSTERED_BAD_CHANNELS",
            "num_recorded_channels": len(recorded_chs),
            "bad_channels_detected": bad_chs,
            "type": "macro_30s"
        } for idx in range(num_windows)]
        return np.stack(slices, axis=0), metadata, raw_stages

    data_uV = raw_clean_eeg.get_data() * 1e6
    data_clipped = np.clip(data_uV, -500.0, 500.0)
    data_harmonized = harmonize_channels_to_cbramod(data_clipped, raw_clean_eeg.ch_names)

    slices = []
    metadata = []
    stages = []

    for idx in range(num_windows):
        start_sample = idx * samples_per_window
        end_sample = start_sample + samples_per_window

        window_raw = data_harmonized[:, start_sample:end_sample]
        window_norm, is_valid, quality_status = process_and_normalize_slice(window_raw)

        slices.append(window_norm)
        stages.append(raw_stages[idx])

        metadata.append({
            "window_idx": idx,
            "start_sec": start_sample / sfreq,
            "end_sec": end_sample / sfreq,
            "samples": samples_per_window,
            "is_valid": is_valid,
            "quality_status": quality_status,
            "num_recorded_channels": len(recorded_chs),
            "bad_channels_interpolated": bad_chs,
            "type": "macro_30s"
        })

    if len(slices) == 0:
        return np.empty((0, 64, samples_per_window)), [], []

    tensor = np.stack(slices, axis=0)
    return tensor, metadata, stages


def slice_strategy_b_micro(
    raw_clean_eeg: mne.io.BaseRaw,
    recorded_chs: List[str],
    bad_chs: List[str],
    subject_ok: bool,
    crop_duration_sec: float = 3.0, 
    spindle_band: Tuple[float, float] = (11.0, 16.0),
    threshold_std: float = 1.5
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Strategy B: Event-centered candidate spindle crops using pre-cleaned continuous EEG.
    Uses Hilbert transform envelope extraction on narrow spindle band without redundant high-pass filtering.
    """
    if not subject_ok:
        return np.empty((0, 64, int(crop_duration_sec * raw_clean_eeg.info["sfreq"]))), [], []

    sfreq = raw_clean_eeg.info["sfreq"]
    crop_samples = int(round(crop_duration_sec * sfreq))
    half_crop = crop_samples // 2

    data_uV = raw_clean_eeg.get_data() * 1e6
    data_clipped = np.clip(data_uV, -500.0, 500.0)
    data_harmonized = harmonize_channels_to_cbramod(data_clipped, raw_clean_eeg.ch_names)

    candidate_crops = []
    meta_candidates = []

    if HAS_YASA:
        try:
            sp = yasa.spindles_detect(raw_clean_eeg, verbose=False)
            if sp is not None:
                summary = sp.summary()
                for _, row in summary.iterrows():
                    peak_sec = row["Peak"]
                    peak_sample = int(round(peak_sec * sfreq))

                    start_s = peak_sample - half_crop
                    end_s = peak_sample + half_crop

                    if start_s >= 0 and end_s <= data_harmonized.shape[1]:
                        crop_data = data_harmonized[:, start_s:end_s]
                        if crop_data.shape[1] == crop_samples:
                            candidate_crops.append(crop_data)
                            meta_candidates.append({
                                "event_channel": row["Channel"],
                                "peak_sec": peak_sec,
                                "duration_sec": row["Duration"],
                                "frequency": row["Frequency"],
                                "num_recorded_channels": len(recorded_chs),
                                "bad_channels_interpolated": bad_chs,
                                "type": "event_crop_yasa"
                            })
        except Exception:
            pass

    # Fallback: Hilbert envelope thresholding (avoids double filtering)
    if len(candidate_crops) == 0:
        # Bandpass filter specifically for spindle band envelope extraction
        raw_spindle = raw_clean_eeg.copy().filter(
            l_freq=spindle_band[0], h_freq=spindle_band[1], verbose=False
        )
        spindle_data = raw_spindle.get_data()
        
        # Hilbert envelope calculation directly on bandpassed signal
        envelope = np.abs(hilbert(spindle_data, axis=-1))
        mean_env = np.mean(envelope)
        std_env = np.std(envelope)
        threshold = mean_env + (threshold_std * std_env)

        peak_indices = np.where(np.max(envelope, axis=0) > threshold)[0]

        refractory_samples = int(1.5 * sfreq)
        selected_peaks = []
        last_p = -refractory_samples
        for p in peak_indices:
            if p - last_p >= refractory_samples:
                if p - half_crop >= 0 and p + half_crop <= data_harmonized.shape[1]:
                    selected_peaks.append(p)
                    last_p = p

        for idx, peak in enumerate(selected_peaks):
            start_s = peak - half_crop
            end_s = peak + half_crop
            crop_data = data_harmonized[:, start_s:end_s]
            if crop_data.shape[1] == crop_samples:
                candidate_crops.append(crop_data)
                meta_candidates.append({
                    "event_idx": idx,
                    "peak_sec": peak / sfreq,
                    "duration_sec": crop_duration_sec,
                    "num_recorded_channels": len(recorded_chs),
                    "bad_channels_interpolated": bad_chs,
                    "type": "event_crop_bandpass"
                })

    slices = []
    metadata = []
    valid_stages = []

    for crop, meta in zip(candidate_crops, meta_candidates):
        norm_crop, is_valid, quality_status = process_and_normalize_slice(crop)
        if is_valid:
            slices.append(norm_crop)
            meta["is_valid"] = True
            meta["quality_status"] = quality_status
            metadata.append(meta)
            valid_stages.append("EVENT")

    if len(slices) == 0:
        return np.empty((0, 64, crop_samples)), [], []

    tensor = np.stack(slices, axis=0)
    return tensor, metadata, valid_stages


def process_subject_slicing_worker(
    args_tuple: Tuple[Path, Path, re.Pattern, str, float, bool]
) -> Dict[str, Union[str, int]]:
    """Worker task executing top-level continuous cleansing and slicing per subject."""
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
        # Step 1: Pre-process, filter (0.3-35 Hz), reference, validate spatial topology, & interpolate
        raw_orig, raw_clean_eeg, recorded_chs, bad_chs, subject_ok = prepare_clean_raw_eeg(src_file)

        # Step 2: Dispatch pre-cleaned object to slicing strategy
        if strategy.lower() == "macro":
            tensor, meta, stages = slice_strategy_a_macro(
                raw_orig, raw_clean_eeg, recorded_chs, bad_chs, subject_ok, window_sec=window_sec
            )
        elif strategy.lower() == "micro":
            tensor, meta, stages = slice_strategy_b_micro(
                raw_clean_eeg, recorded_chs, bad_chs, subject_ok, crop_duration_sec=window_sec
            )
        else:
            raise ValueError(f"Unknown slicing strategy: {strategy}")

        # Step 3: Save binary matrix [num_windows, 64, samples] and metadata JSON
        np.save(output_npy, tensor)
        with open(output_meta, "w") as f:
            json.dump({
                "subject_id": subject_id,
                "strategy": strategy,
                "num_channels": 64,
                "channel_names": CBRMOD_STANDARD_64,
                "sampling_freq": raw_clean_eeg.info["sfreq"],
                "num_slices": len(meta),
                "stages": stages,
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
    parser = argparse.ArgumentParser(
        description="EEG Slicing Pipeline with Top-Level Continuous Filtering & Neighborhood Quality Tracking"
    )
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