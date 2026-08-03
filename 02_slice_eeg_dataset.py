#!/usr/bin/env python3
"""
EEG Window Slicing Pipeline with Continuous Preprocessing, Spatial Validation, 
Spherical Spline Interpolation, and Full Debug Logging for Channel & Window Rejections.

Architecture Pipeline:
1. Load Raw File -> Read continuous time-series recording (.edf, .fif, .bdf, .vhdr).
2. Dynamic Discovery -> Identify recorded scalp channels matching CBraMod layout at runtime.
3. Channel Selection -> Crop away non-EEG/trigger channels using `inst.pick()`.
4. Continuous Filtering -> Apply single-pass 0.3-35.0 Hz zero-phase FIR bandpass filter globally.
5. Referencing -> Apply linked-earlobe (A1/A2) or CAR referencing.
6. Neighborhood Spatial Validation & Interpolation (LOGGED WITH SUBJ ID) -> 
   Detect flatlines/high-variance noise, check spatial adjacency (reject spatial clusters of bad channels), 
   and interpolate isolated bad channels via spherical splines.
7. CBraMod Harmonization -> Map cleaned physical signals into 64-channel matrix; zero-pad unrecorded positions (0.0 uV).
8. Window-Level Slicing, Normalization & Quality Screening (LOGGED WITH SUBJ ID) -> 
   Epoch continuous data using `raw.n_times`, evaluate active channels for residual window artifacts/flatlines with detailed 
   debug logs tagged by subject ID, and apply per-window Z-score normalization.
"""

import argparse
import json
import logging
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

logger = logging.getLogger("EEG_Pipeline")


def configure_logging(verbose: bool = False):
    """Configures process-level logging format and severity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )


def clean_ch_name(name: str) -> str:
    """Standardizes channel naming for robust string matching."""
    return name.upper().replace('.', '').replace('-', '').replace('EEG', '').strip()


def discover_cbramod_channels(raw: mne.io.BaseRaw) -> List[str]:
    """Dynamically discovers recorded scalp channels that correspond to CBraMod layout."""
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


def apply_eeg_referencing(raw: mne.io.BaseRaw, subject_id: str = "") -> mne.io.BaseRaw:
    """Applies linked-earlobe referencing (A1/A2) if present, else CAR."""
    ch_clean = [clean_ch_name(ch) for ch in raw.ch_names]
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    if "A1" in ch_clean and "A2" in ch_clean:
        a1_ch = raw.ch_names[ch_clean.index("A1")]
        a2_ch = raw.ch_names[ch_clean.index("A2")]
        raw.set_eeg_reference(ref_channels=[a1_ch, a2_ch], projection=False, verbose=False)
        logger.debug(f"{tag}Referencing: Applied Linked-Earlobe (ref={a1_ch},{a2_ch}).")
    else:
        try:
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
            logger.debug(f"{tag}Referencing: Applied Common Average Reference (CAR).")
        except Exception as e:
            logger.warning(f"{tag}Referencing warning: {e}")
    return raw


def validate_spatial_neighborhood(
    raw: mne.io.BaseRaw, 
    bad_ch_names: List[str], 
    max_bad_neighbors: int = 1,
    subject_id: str = ""
) -> bool:
    """Evaluates physical adjacency of bad channels using the 10-20 topology matrix."""
    if len(bad_ch_names) <= 1:
        return True

    tag = f"[Subj: {subject_id}] " if subject_id else ""
    try:
        adjacency_matrix, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type='eeg')
        adj_dense = adjacency_matrix.toarray()
        
        ch_map = {name: idx for idx, name in enumerate(ch_names)}
        bad_indices = [ch_map[ch] for ch in bad_ch_names if ch in ch_map]

        for bad_idx in bad_indices:
            bad_ch_name = ch_names[bad_idx]
            neighbor_indices = np.where(adj_dense[bad_idx])[0]
            bad_neighbors = [ch_names[i] for i in neighbor_indices if i in bad_indices]
            
            if len(bad_neighbors) > max_bad_neighbors:
                logger.warning(
                    f"{tag}[REJECT - SPATIAL CLUSTER] Bad channel '{bad_ch_name}' has {len(bad_neighbors)} bad "
                    f"adjacent neighbors ({bad_neighbors}), exceeding limit of {max_bad_neighbors}."
                )
                return False
                
        logger.debug(f"{tag}Spatial Neighborhood Check PASSED: All bad channels are spatially isolated.")
        return True
    except Exception as e:
        logger.debug(f"{tag}Adjacency check skipped/failed ({e}); assuming spatially safe.")
        return True


def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    recorded_ch_names: List[str],
    min_std_uV: float = 0.5,
    max_std_uV: float = 150.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1,
    subject_id: str = ""
) -> Tuple[mne.io.BaseRaw, List[str], bool]:
    """Evaluates channel quality, spatial neighborhood adjacency, and interpolates bad channels."""
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    n_recorded = len(recorded_ch_names)
    if n_recorded == 0:
        logger.error(f"{tag}No valid recorded EEG channels found.")
        return raw, [], False

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    data_uV = raw.get_data() * 1e6
    ch_stds = np.std(data_uV, axis=-1)

    bad_mask = (ch_stds < min_std_uV) | (ch_stds > max_std_uV)
    bad_indices = np.where(bad_mask)[0]
    bad_ch_names = [raw.ch_names[i] for i in bad_indices]

    bad_count = len(bad_ch_names)
    bad_ratio = bad_count / float(n_recorded)

    if bad_count > 0:
        logger.debug(f"{tag}Detected {bad_count}/{n_recorded} suspicious channels ({bad_ratio:.1%}): {bad_ch_names}")
        for i in bad_indices:
            val = ch_stds[i]
            reason = "FLATLINE" if val < min_std_uV else "HIGH_VARIANCE/NOISE"
            logger.debug(f"{tag}  └─ Bad Channel '{raw.ch_names[i]}': std = {val:.2f} uV ({reason})")

    # 1. Reject if total bad channel ratio exceeds limit
    if bad_ratio > max_bad_ratio:
        logger.warning(
            f"{tag}[REJECT - HIGH BAD RATIO] Subject has {bad_count}/{n_recorded} bad channels "
            f"({bad_ratio:.1%}), exceeding limit of {max_bad_ratio:.1%}."
        )
        return raw, bad_ch_names, False

    # 2. Reject if bad channels form spatial clusters
    if bad_count > 1:
        is_spatially_isolated = validate_spatial_neighborhood(
            raw, bad_ch_names, max_bad_neighbors=max_bad_neighbors, subject_id=subject_id
        )
        if not is_spatially_isolated:
            return raw, bad_ch_names, False

    # 3. Interpolate isolated bad channels
    if bad_count > 0:
        clean_count = n_recorded - bad_count
        logger.info(
            f"{tag}[INTERPOLATION] Reconstructing {bad_count} isolated bad channels {bad_ch_names} "
            f"using {clean_count} clean spatial anchors via spherical splines."
        )
        raw.info['bads'] = bad_ch_names
        raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
        logger.debug(f"{tag}Interpolation successfully applied.")
    else:
        logger.debug(f"{tag}No bad channels detected. All recorded channels are clean.")

    return raw, bad_ch_names, True


def prepare_clean_raw_eeg(
    file_path: Path, 
    subject_id: str = "",
    l_freq: float = 0.3, 
    h_freq: float = 35.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, List[str], List[str], bool]:
    """Top-level continuous data preparation pipeline."""
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    logger.debug(f"{tag}Processing raw file: {file_path.name}")
    raw_orig = load_raw_eeg(file_path)

    recorded_eeg_chs = discover_cbramod_channels(raw_orig)
    if len(recorded_eeg_chs) == 0:
        logger.error(f"{tag}Zero matching CBraMod channels found in {file_path.name}")
        return raw_orig, raw_orig, [], [], False

    logger.debug(f"{tag}Discovered {len(recorded_eeg_chs)} valid scalp EEG channels.")

    raw_eeg = raw_orig.copy().pick(recorded_eeg_chs)

    logger.debug(f"{tag}Applying zero-phase FIR bandpass filter ({l_freq} Hz - {h_freq} Hz)...")
    raw_eeg.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )

    raw_ref = apply_eeg_referencing(raw_eeg, subject_id=subject_id)

    raw_clean_eeg, bad_chs, subject_ok = detect_and_interpolate_bad_channels(
        raw_ref, 
        recorded_ch_names=recorded_eeg_chs, 
        max_bad_ratio=max_bad_ratio,
        max_bad_neighbors=max_bad_neighbors,
        subject_id=subject_id
    )

    return raw_orig, raw_clean_eeg, recorded_eeg_chs, bad_chs, subject_ok


def harmonize_channels_to_cbramod(data_uV: np.ndarray, orig_ch_names: List[str]) -> np.ndarray:
    """Maps cleaned physical channels into CBraMod's standard 64-channel matrix (zero-padded)."""
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
    slice_64: np.ndarray, 
    min_std: float = 0.5, 
    max_std: float = 150.0,
    window_id: str = "",
    subject_id: str = ""
) -> Tuple[np.ndarray, bool, str]:
    """
    Evaluates window active channels for residual flatlines or extreme artifacts with 
    detailed debug logging, then applies per-window Z-score normalization strictly on active channels.
    """
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    nonzero_mask = np.abs(slice_64).sum(axis=-1) > 1e-8

    if not np.any(nonzero_mask):
        logger.debug(f"{tag}[REJECT WINDOW {window_id}] Reason: EMPTY_CHANNEL_DATA (All channels zero-padded/empty).")
        return np.zeros_like(slice_64, dtype=np.float32), False, "EMPTY_CHANNEL_DATA"

    active_stds = slice_64[nonzero_mask].std(axis=-1)

    # 1. Check for residual channel flatlining within window
    min_found_std = active_stds.min()
    if min_found_std < min_std:
        logger.debug(
            f"{tag}[REJECT WINDOW {window_id}] Reason: FLATLINE_DETECTED "
            f"(Min active channel std: {min_found_std:.3f} uV < threshold {min_std:.2f} uV)."
        )
        return np.zeros_like(slice_64, dtype=np.float32), False, "FLATLINE_DETECTED"

    # 2. Check for extreme transient artifacts (e.g., electrode pop or motion)
    max_found_std = active_stds.max()
    if max_found_std > max_std:
        logger.debug(
            f"{tag}[REJECT WINDOW {window_id}] Reason: EXTREME_ARTIFACT "
            f"(Max active channel std: {max_found_std:.2f} uV > threshold {max_std:.2f} uV)."
        )
        return np.zeros_like(slice_64, dtype=np.float32), False, "EXTREME_ARTIFACT"

    # 3. Apply Z-score normalization strictly across active channels
    normalized = slice_64.copy()
    means = normalized[nonzero_mask].mean(axis=-1, keepdims=True)
    stds = normalized[nonzero_mask].std(axis=-1, keepdims=True)

    normalized[nonzero_mask] = (normalized[nonzero_mask] - means) / (stds + 1e-8)
    return normalized, True, "OK"


def extract_epoch_stages(raw: mne.io.BaseRaw, num_windows: int, window_sec: float = 30.0) -> List[str]:
    """Parses annotations for sleep stage labels."""
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
    """Selects central channel for YASA."""
    preferences = ["C4", "C3", "CZ"]
    for pref in preferences:
        pattern = re.compile(rf"\b{pref}\b", re.IGNORECASE)
        for ch in eeg_chs:
            if pattern.search(ch):
                return ch
    return eeg_chs[0]


def predict_epoch_stages_yasa(raw: mne.io.BaseRaw, num_windows: int) -> List[str]:
    """YASA fallback predictor."""
    try:
        eeg_chs = raw.copy().pick('eeg').ch_names
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
    window_sec: float = 30.0,
    subject_id: str = ""
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Strategy A: 30-second continuous macro window slicing with detailed quality logging."""
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    sfreq = raw_clean_eeg.info["sfreq"]
    samples_per_window = int(round(window_sec * sfreq))
    total_samples = raw_clean_eeg.n_times
    num_windows = total_samples // samples_per_window

    if len(raw_orig.annotations) > 0:
        raw_stages = extract_epoch_stages(raw_orig, num_windows=num_windows, window_sec=window_sec)
    else:
        raw_stages = predict_epoch_stages_yasa(raw_clean_eeg, num_windows=num_windows)

    if not subject_ok:
        logger.warning(f"{tag}[STRATEGY A] Recording unrecoverable. Zeroing all {num_windows} macro windows.")
        slices = [np.zeros((64, samples_per_window), dtype=np.float32) for _ in range(num_windows)]
        metadata = [{
            "window_idx": idx,
            "is_valid": False,
            "quality_status": "REJECTED_BAD_CHANNELS_OR_CLUSTERS",
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
    rejection_reasons = {}

    for idx in range(num_windows):
        start_sample = idx * samples_per_window
        end_sample = start_sample + samples_per_window
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        window_raw = data_harmonized[:, start_sample:end_sample]
        
        # Screen window quality and apply Z-score normalization
        win_id_str = f"#{idx} ({start_sec:.1f}s-{end_sec:.1f}s)"
        window_norm, is_valid, quality_status = process_and_normalize_slice(
            window_raw, window_id=win_id_str, subject_id=subject_id
        )

        if not is_valid:
            rejection_reasons[quality_status] = rejection_reasons.get(quality_status, 0) + 1

        slices.append(window_norm)
        stages.append(raw_stages[idx])

        metadata.append({
            "window_idx": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "samples": samples_per_window,
            "is_valid": is_valid,
            "quality_status": quality_status,
            "num_recorded_channels": len(recorded_chs),
            "bad_channels_interpolated": bad_chs,
            "type": "macro_30s"
        })

    valid_count = sum(1 for m in metadata if m["is_valid"])
    rejected_count = num_windows - valid_count
    
    logger.info(
        f"{tag}[STRATEGY A SUMMARY] Total: {num_windows} windows | Valid: {valid_count} | "
        f"Rejected: {rejected_count} ({(rejected_count/num_windows)*100:.1f}%). "
        f"Rejection breakdown: {rejection_reasons if rejection_reasons else 'None'}"
    )

    if len(slices) == 0:
        return np.empty((0, 64, samples_per_window)), [], []

    return np.stack(slices, axis=0), metadata, stages


def slice_strategy_b_micro(
    raw_clean_eeg: mne.io.BaseRaw,
    recorded_chs: List[str],
    bad_chs: List[str],
    subject_ok: bool,
    crop_duration_sec: float = 3.0, 
    spindle_band: Tuple[float, float] = (11.0, 16.0),
    threshold_std: float = 1.5,
    subject_id: str = ""
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Strategy B: Event-centered candidate spindle crops with window quality logging."""
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    if not subject_ok:
        logger.warning(f"{tag}[STRATEGY B] Recording unrecoverable; skipping candidate event extraction.")
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

    # Fallback: Hilbert envelope thresholding on spindle band
    if len(candidate_crops) == 0:
        raw_spindle = raw_clean_eeg.copy().filter(
            l_freq=spindle_band[0], h_freq=spindle_band[1], verbose=False
        )
        spindle_data = raw_spindle.get_data()
        
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
    rejection_reasons = {}

    for idx, (crop, meta) in enumerate(zip(candidate_crops, meta_candidates)):
        peak_t = meta.get("peak_sec", 0.0)
        win_id_str = f"event_{idx} (peak={peak_t:.1f}s)"
        
        norm_crop, is_valid, quality_status = process_and_normalize_slice(
            crop, window_id=win_id_str, subject_id=subject_id
        )
        
        if is_valid:
            slices.append(norm_crop)
            meta["is_valid"] = True
            meta["quality_status"] = quality_status
            metadata.append(meta)
            valid_stages.append("EVENT")
        else:
            rejection_reasons[quality_status] = rejection_reasons.get(quality_status, 0) + 1

    total_candidates = len(candidate_crops)
    valid_count = len(slices)
    rejected_count = total_candidates - valid_count

    logger.info(
        f"{tag}[STRATEGY B SUMMARY] Extracted candidates: {total_candidates} | Valid: {valid_count} | "
        f"Rejected: {rejected_count}. Rejection breakdown: {rejection_reasons if rejection_reasons else 'None'}"
    )

    if len(slices) == 0:
        return np.empty((0, 64, crop_samples)), [], []

    return np.stack(slices, axis=0), metadata, valid_stages


def process_subject_slicing_worker(
    args_tuple: Tuple[Path, Path, re.Pattern, str, float, bool, bool]
) -> Dict[str, Union[str, int]]:
    """Worker task executing preprocessing and slicing per subject."""
    src_file, dst_dir, pattern, strategy, window_sec, force, verbose = args_tuple

    configure_logging(verbose=verbose)

    # Derive subject / file ID upfront for logging context
    subject_id = src_file.stem
    if pattern:
        match = pattern.search(subject_id)
        if match:
            subject_id = match.group(0)

    output_npy = dst_dir / f"{subject_id}_windows.npy"
    output_meta = dst_dir / f"{subject_id}_meta.json"

    dst_dir.mkdir(parents=True, exist_ok=True)

    if output_npy.exists() and output_meta.exists() and not force:
        logger.debug(f"[Subj: {subject_id}] Output already exists; skipping.")
        return {"status": "SKIPPED", "subject": subject_id, "count": 0}

    try:
        raw_orig, raw_clean_eeg, recorded_chs, bad_chs, subject_ok = prepare_clean_raw_eeg(
            src_file, subject_id=subject_id
        )

        if strategy.lower() == "macro":
            tensor, meta, stages = slice_strategy_a_macro(
                raw_orig, raw_clean_eeg, recorded_chs, bad_chs, subject_ok, window_sec=window_sec, subject_id=subject_id
            )
        elif strategy.lower() == "micro":
            tensor, meta, stages = slice_strategy_b_micro(
                raw_clean_eeg, recorded_chs, bad_chs, subject_ok, crop_duration_sec=window_sec, subject_id=subject_id
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
        logger.error(f"[Subj: {subject_id}] Error processing file: {str(e)}", exc_info=True)
        return {"status": f"ERROR: {str(e)}", "subject": subject_id, "count": 0}


def run_slicing_pipeline(
    src_dir: Path,
    dst_dir: Path,
    pattern: re.Pattern = None,
    strategy: str = "macro",
    window_sec: float = 30.0,
    num_workers: int = 1,
    force: bool = False,
    verbose: bool = False
):
    """Executes parallel subject-level window slicing."""
    configure_logging(verbose=verbose)

    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))
    files = sorted(files)

    print(f"Found {len(files)} files for slicing. Strategy: '{strategy.upper()}', Window: {window_sec}s")

    tasks = [
        (f, dst_dir / f.relative_to(src_dir).parent, pattern, strategy, window_sec, force, verbose) 
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
        description="EEG Slicing Pipeline with Debug Logging for Interpolation & Window Quality Screening"
    )
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing resampled files")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output destination directory")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject id")
    parser.add_argument("--strategy", type=str, choices=["macro", "micro"], default="macro", help="Slicing strategy")
    parser.add_argument("--window_sec", type=float, default=30.0, help="Window duration in seconds")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="CPU worker count")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging")

    args = parser.parse_args()

    run_slicing_pipeline(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        pattern=args.pattern,
        strategy=args.strategy,
        window_sec=args.window_sec,
        num_workers=args.num_workers,
        force=args.force,
        verbose=args.verbose
    )