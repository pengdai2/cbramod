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
import yasa
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
from cbramod_utils import find_eeg_files, load_raw_eeg, valid_regex
from cbramod_utils import extract_epoch_stages, predict_epoch_stages_yasa


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
        # No try/except swallow here on purpose: a referencing failure must fail loud (propagate to
        # process_subject_slicing_worker's outer try/except, which marks the subject ERROR) rather
        # than silently returning `raw` unreferenced with zero trace in the output metadata that
        # referencing never actually happened.
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
        logger.debug(f"{tag}Referencing: Applied Common Average Reference (CAR).")
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
        # Fail CLOSED, not open: if the adjacency check itself can't run, we have no way to verify
        # this subject's bad channels aren't spatially clustered -- and the whole point of this check
        # (per this file's own docstring) is to REJECT spatially-clustered bad channels rather than
        # interpolate them. Silently assuming "safe" here would let exactly the subjects this check
        # exists to catch slip through with no record that the check never actually ran.
        logger.warning(f"{tag}[REJECT - ADJACENCY CHECK FAILED] Could not verify spatial isolation ({e}); "
                        f"rejecting rather than assuming safe.")
        return False


def standardize_mne_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Renames raw channels to match standard MNE 10-20 casing (e.g., 'FP1' -> 'Fp1', 'CZ' -> 'Cz')."""
    montage_1020 = mne.channels.make_standard_montage("standard_1020")
    montage_map = {clean_ch_name(ch): ch for ch in montage_1020.ch_names}
    
    rename_dict = {}
    for ch in raw.ch_names:
        cleaned = clean_ch_name(ch)
        if cleaned in montage_map and ch != montage_map[cleaned]:
            rename_dict[ch] = montage_map[cleaned]
            
    if rename_dict:
        raw.rename_channels(rename_dict)
    return raw


def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    recorded_ch_names: List[str],
    min_std_uV: float = 0.5,
    max_std_uV: float = 150.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1,
    subject_id: str = ""
) -> Tuple[mne.io.BaseRaw, List[str], List[str], bool]:
    """
    Evaluates channel quality, spatial neighborhood adjacency, and interpolates bad channels safely.

    Returns (raw, interpolated_chs, bad_not_interpolated_chs, subject_ok) -- interpolated_chs is
    exactly the channels that were successfully spatial-spline-reconstructed; bad_not_interpolated_chs
    covers every OTHER flagged-bad channel (missing 3D coordinates, or detected but never attempted
    because the subject was rejected outright for a high bad-ratio or spatially-clustered bad
    channels). Callers must not treat bad_not_interpolated_chs as cleaned -- its data is whatever the
    original flatline/noisy/uninterpolated signal was.
    """
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    n_recorded = len(recorded_ch_names)
    if n_recorded == 0:
        logger.error(f"{tag}No valid recorded EEG channels found.")
        return raw, [], [], False

    # 1. Standardize channel casing to match MNE's standard 10-20 montage
    raw = standardize_mne_channel_names(raw)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # 2. Audit channels for valid 3D coordinates (non-NaN / non-zero)
    valid_pos_chs = set()
    for ch in raw.info['chs']:
        loc = ch['loc'][:3]
        if not np.isnan(loc).any() and not np.all(loc == 0):
            valid_pos_chs.add(ch['ch_name'])
        else:
            logger.warning(f"{tag}Channel '{ch['ch_name']}' lacks valid 3D spatial coordinates (NaN location).")

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

    # Reject if total bad channel ratio exceeds limit -- none of the bad channels were ever attempted,
    # so they all land in bad_not_interpolated_chs, never in interpolated_chs.
    if bad_ratio > max_bad_ratio:
        logger.warning(
            f"{tag}[REJECT - HIGH BAD RATIO] Subject has {bad_count}/{n_recorded} bad channels "
            f"({bad_ratio:.1%}), exceeding limit of {max_bad_ratio:.1%}."
        )
        return raw, [], bad_ch_names, False

    # Reject if bad channels form spatial clusters -- same rationale, nothing was interpolated.
    if bad_count > 1:
        is_spatially_isolated = validate_spatial_neighborhood(
            raw, bad_ch_names, max_bad_neighbors=max_bad_neighbors, subject_id=subject_id
        )
        if not is_spatially_isolated:
            return raw, [], bad_ch_names, False

    # 3. Interpolate isolated bad channels with coordinate safety check
    interpolated_chs: List[str] = []
    bad_not_interpolated_chs: List[str] = []
    if bad_count > 0:
        # Keep only bad channels that have valid 3D spatial locations
        interpolatable_bads = [ch for ch in bad_ch_names if ch in valid_pos_chs]
        skipped_no_coords = [ch for ch in bad_ch_names if ch not in valid_pos_chs]

        if skipped_no_coords:
            logger.warning(f"{tag}Skipping interpolation for bad channels missing 3D coordinates: {skipped_no_coords}")
        bad_not_interpolated_chs.extend(skipped_no_coords)

        if interpolatable_bads:
            clean_count = n_recorded - len(interpolatable_bads)
            logger.info(
                f"{tag}[INTERPOLATION] Reconstructing {len(interpolatable_bads)} isolated bad channels {interpolatable_bads} "
                f"using {clean_count} clean spatial anchors via spherical splines."
            )
            raw.info['bads'] = interpolatable_bads
            try:
                raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
                logger.debug(f"{tag}Interpolation successfully applied.")
                interpolated_chs = list(interpolatable_bads)
            except Exception as e:
                logger.error(
                    f"{tag}[REJECT - INTERPOLATION FAILURE] Spherical spline interpolation failed: {e}. "
                    f"Marking subject as unrecoverable."
                )
                return raw, [], bad_ch_names, False
    else:
        logger.debug(f"{tag}No bad channels detected. All recorded channels are clean.")

    return raw, interpolated_chs, bad_not_interpolated_chs, True


def prepare_clean_raw_eeg(
    file_path: Path, 
    subject_id: str = "",
    l_freq: float = 0.3, 
    h_freq: float = 35.0,
    max_bad_ratio: float = 0.15,
    max_bad_neighbors: int = 1
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, List[str], List[str], List[str], bool]:
    """
    Top-level continuous data preparation pipeline.

    Returns (raw_orig, raw_clean_eeg, recorded_eeg_chs, interpolated_chs, bad_not_interpolated_chs,
    subject_ok) -- see detect_and_interpolate_bad_channels()'s docstring for what interpolated_chs vs
    bad_not_interpolated_chs mean.
    """
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    logger.debug(f"{tag}Processing raw file: {file_path.name}")
    raw_orig = load_raw_eeg(file_path)

    recorded_eeg_chs = discover_cbramod_channels(raw_orig)
    if len(recorded_eeg_chs) == 0:
        logger.error(f"{tag}Zero matching CBraMod channels found in {file_path.name}")
        return raw_orig, raw_orig, [], [], [], False

    logger.debug(f"{tag}Discovered {len(recorded_eeg_chs)} valid scalp EEG channels.")

    # Locate A1/A2 (linked-earlobe reference electrodes) BEFORE picking down to the CBraMod scalp
    # layout. A1/A2 aren't part of CBRMOD_STANDARD_64, so discover_cbramod_channels() never finds
    # them -- picking only recorded_eeg_chs here (as this used to do unconditionally) would silently
    # drop A1/A2 before apply_eeg_referencing() ever runs, making its "prefer linked-earlobe if
    # present" check permanently unreachable and forcing every subject onto Common Average Reference
    # regardless of whether A1/A2 actually exist in the recording.
    ch_clean_orig = [clean_ch_name(ch) for ch in raw_orig.ch_names]
    earlobe_chs = [raw_orig.ch_names[ch_clean_orig.index(name)] for name in ("A1", "A2") if name in ch_clean_orig]
    if earlobe_chs:
        logger.debug(f"{tag}Found linked-earlobe reference channel(s): {earlobe_chs}.")

    pick_chs = recorded_eeg_chs + [ch for ch in earlobe_chs if ch not in recorded_eeg_chs]
    raw_eeg = raw_orig.copy().pick(pick_chs)

    logger.debug(f"{tag}Applying zero-phase FIR bandpass filter ({l_freq} Hz - {h_freq} Hz)...")
    raw_eeg.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )

    raw_ref = apply_eeg_referencing(raw_eeg, subject_id=subject_id)

    # Drop A1/A2 now that referencing is done -- they were only needed as the reference and aren't
    # part of the CBraMod 64-channel scalp layout the rest of the pipeline (bad-channel detection,
    # interpolation, harmonization) operates on.
    if earlobe_chs:
        raw_ref = raw_ref.copy().pick(recorded_eeg_chs)

    raw_clean_eeg, interpolated_chs, bad_not_interpolated_chs, subject_ok = detect_and_interpolate_bad_channels(
        raw_ref,
        recorded_ch_names=recorded_eeg_chs,
        max_bad_ratio=max_bad_ratio,
        max_bad_neighbors=max_bad_neighbors,
        subject_id=subject_id
    )

    return raw_orig, raw_clean_eeg, recorded_eeg_chs, interpolated_chs, bad_not_interpolated_chs, subject_ok


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


def compute_active_channel_mask(orig_ch_names: List[str]) -> List[bool]:
    """
    Boolean mask, aligned with CBRMOD_STANDARD_64 (and thus with `channel_names` in the output
    meta.json and every axis-0 row of the harmonized 64-channel tensor), True where that position was
    actually recorded for this subject and False where harmonize_channels_to_cbramod() zero-padded it.

    Persisting this explicitly (rather than leaving downstream consumers to infer channel activity
    from e.g. norm_std_uv > 0, which is only valid for is_valid=True windows and is a repurposed
    byproduct of normalization rather than an intentional flag) is what several analysis scripts this
    session ended up needing to reverse-engineer from scratch, once, each -- this makes that a
    non-issue going forward.
    """
    clean_orig = set(clean_ch_name(ch) for ch in orig_ch_names)
    return [clean_ch_name(ch) in clean_orig for ch in CBRMOD_STANDARD_64]


def process_and_normalize_slice(
    slice_64: np.ndarray,
    min_std: float = 0.5,
    max_std: float = 150.0,
    window_id: str = "",
    subject_id: str = ""
) -> Tuple[np.ndarray, bool, str, np.ndarray, np.ndarray]:
    """
    Evaluates window active channels for residual flatlines or extreme artifacts with
    detailed debug logging, then applies per-window Z-score normalization strictly on active channels.

    Also returns the per-channel mean/std (uV) actually used for that
    normalization (`norm_mean_uv`/`norm_std_uv`, length == slice_64.shape[0],
    0.0 for zero-padded/inactive channels and for rejected windows, which
    are never normalized at all). Callers should persist these into the
    slice's metadata so the exact original uV signal can be reconstructed
    later without needing to re-derive anything from the source recording:

        original_uV = normalized * (norm_std_uv + 1e-8) + norm_mean_uv

    (same 1e-8 epsilon as used below, required for an exact inverse).
    """
    num_channels = slice_64.shape[0]
    zero_stats = np.zeros(num_channels, dtype=np.float32)
    tag = f"[Subj: {subject_id}] " if subject_id else ""
    nonzero_mask = np.abs(slice_64).sum(axis=-1) > 1e-8

    if not np.any(nonzero_mask):
        logger.debug(f"{tag}[REJECT WINDOW {window_id}] Reason: EMPTY_CHANNEL_DATA (All channels zero-padded/empty).")
        return np.zeros_like(slice_64, dtype=np.float32), False, "EMPTY_CHANNEL_DATA", zero_stats.copy(), zero_stats.copy()

    active_stds = slice_64[nonzero_mask].std(axis=-1)

    # 1. Check for residual channel flatlining within window
    min_found_std = active_stds.min()
    if min_found_std < min_std:
        logger.debug(
            f"{tag}[REJECT WINDOW {window_id}] Reason: FLATLINE_DETECTED "
            f"(Min active channel std: {min_found_std:.3f} uV < threshold {min_std:.2f} uV)."
        )
        return np.zeros_like(slice_64, dtype=np.float32), False, "FLATLINE_DETECTED", zero_stats.copy(), zero_stats.copy()

    # 2. Check for extreme transient artifacts (e.g., electrode pop or motion)
    max_found_std = active_stds.max()
    if max_found_std > max_std:
        logger.debug(
            f"{tag}[REJECT WINDOW {window_id}] Reason: EXTREME_ARTIFACT "
            f"(Max active channel std: {max_found_std:.2f} uV > threshold {max_std:.2f} uV)."
        )
        return np.zeros_like(slice_64, dtype=np.float32), False, "EXTREME_ARTIFACT", zero_stats.copy(), zero_stats.copy()

    # 3. Apply Z-score normalization strictly across active channels
    normalized = slice_64.copy()
    means = normalized[nonzero_mask].mean(axis=-1, keepdims=True)
    stds = normalized[nonzero_mask].std(axis=-1, keepdims=True)

    normalized[nonzero_mask] = (normalized[nonzero_mask] - means) / (stds + 1e-8)

    norm_mean_uv = zero_stats.copy()
    norm_std_uv = zero_stats.copy()
    norm_mean_uv[nonzero_mask] = means.squeeze(-1)
    norm_std_uv[nonzero_mask] = stds.squeeze(-1)

    return normalized, True, "OK", norm_mean_uv, norm_std_uv


def slice_strategy_a_macro(
    raw_orig: mne.io.BaseRaw,
    raw_clean_eeg: mne.io.BaseRaw,
    recorded_chs: List[str],
    interpolated_chs: List[str],
    bad_not_interpolated_chs: List[str],
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
            # Same key names as the valid-window path below (was "bad_channels_detected" here vs
            # "bad_channels_interpolated" there -- any downstream reader keyed on one specific name,
            # e.g. p02x_verify_reslice_diff.py, silently got an empty list for rejected subjects).
            "bad_channels_interpolated": interpolated_chs,
            "bad_channels_not_interpolated": bad_not_interpolated_chs,
            "num_samples_clipped": 0,
            "norm_mean_uv": [0.0] * 64,
            "norm_std_uv": [0.0] * 64,
            "type": "macro_30s"
        } for idx in range(num_windows)]
        return np.stack(slices, axis=0), metadata, raw_stages

    data_uV = raw_clean_eeg.get_data() * 1e6
    # Track which samples the +-500uV clip actually alters, BEFORE clipping -- a clipped flat-top is a
    # broadband transient (spectral leakage into every band, disproportionately the higher ones), so
    # this is a cheap diagnostic to see how often it fires in practice before deciding whether the
    # fixed clip bound needs to become something smarter (e.g. per-segment interpolation).
    was_clipped = np.abs(data_uV) > 500.0
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
        num_samples_clipped = int(was_clipped[:, start_sample:end_sample].sum())

        # Screen window quality and apply Z-score normalization
        win_id_str = f"#{idx} ({start_sec:.1f}s-{end_sec:.1f}s)"
        window_norm, is_valid, quality_status, norm_mean_uv, norm_std_uv = process_and_normalize_slice(
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
            "bad_channels_interpolated": interpolated_chs,
            "bad_channels_not_interpolated": bad_not_interpolated_chs,
            "num_samples_clipped": num_samples_clipped,
            # Per-channel mean/std (uV) used for this window's Z-score normalization --
            # 0.0 for zero-padded/inactive channels or rejected windows (never normalized).
            # Reconstruct the original uV signal via: normalized*(norm_std_uv+1e-8) + norm_mean_uv
            "norm_mean_uv": norm_mean_uv.tolist(),
            "norm_std_uv": norm_std_uv.tolist(),
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
    interpolated_chs: List[str],
    bad_not_interpolated_chs: List[str],
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
    was_clipped = np.abs(data_uV) > 500.0  # see slice_strategy_a_macro's identical comment
    data_clipped = np.clip(data_uV, -500.0, 500.0)
    data_harmonized = harmonize_channels_to_cbramod(data_clipped, raw_clean_eeg.ch_names)

    candidate_crops = []
    meta_candidates = []

    # yasa is imported unconditionally at module level -- no soft-dependency flag to gate this on
    # (an earlier refactor removed the try/except HAS_YASA guard but left this check behind, which
    # referenced an undefined name and made this whole strategy crash immediately on every use).
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
                            "bad_channels_interpolated": interpolated_chs,
                            "bad_channels_not_interpolated": bad_not_interpolated_chs,
                            "num_samples_clipped": int(was_clipped[:, start_s:end_s].sum()),
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
                    "bad_channels_interpolated": interpolated_chs,
                    "bad_channels_not_interpolated": bad_not_interpolated_chs,
                    "num_samples_clipped": int(was_clipped[:, start_s:end_s].sum()),
                    "type": "event_crop_bandpass"
                })

    slices = []
    metadata = []
    valid_stages = []
    rejection_reasons = {}

    for idx, (crop, meta) in enumerate(zip(candidate_crops, meta_candidates)):
        peak_t = meta.get("peak_sec", 0.0)
        win_id_str = f"event_{idx} (peak={peak_t:.1f}s)"

        norm_crop, is_valid, quality_status, norm_mean_uv, norm_std_uv = process_and_normalize_slice(
            crop, window_id=win_id_str, subject_id=subject_id
        )

        if is_valid:
            slices.append(norm_crop)
            # window_idx must be the direct row index into the final .npy array (extract_valid_window_
            # indices() does a required slice_info["window_idx"] lookup, no .get() fallback) -- this
            # was never set at all for Strategy B, so any downstream loader would KeyError on it.
            meta["window_idx"] = len(metadata)
            meta["is_valid"] = True
            meta["quality_status"] = quality_status
            # See process_and_normalize_slice's docstring for the exact inverse transform.
            meta["norm_mean_uv"] = norm_mean_uv.tolist()
            meta["norm_std_uv"] = norm_std_uv.tolist()
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
        raw_orig, raw_clean_eeg, recorded_chs, interpolated_chs, bad_not_interpolated_chs, subject_ok = (
            prepare_clean_raw_eeg(src_file, subject_id=subject_id)
        )

        if strategy.lower() == "macro":
            tensor, meta, stages = slice_strategy_a_macro(
                raw_orig, raw_clean_eeg, recorded_chs, interpolated_chs, bad_not_interpolated_chs,
                subject_ok, window_sec=window_sec, subject_id=subject_id
            )
        elif strategy.lower() == "micro":
            tensor, meta, stages = slice_strategy_b_micro(
                raw_clean_eeg, recorded_chs, interpolated_chs, bad_not_interpolated_chs, subject_ok,
                crop_duration_sec=window_sec, subject_id=subject_id
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Which of the 64 canonical positions were actually recorded for this subject (interpolated
        # channels count as active -- they were fixed, not left zero-padded). Persisted once, subject-
        # level, rather than leaving every downstream consumer to re-derive this from norm_std_uv > 0
        # (only valid for is_valid=True windows, and a repurposed byproduct rather than an intentional
        # flag -- this ambiguity already caused real confusion earlier in this project).
        active_channel_mask = compute_active_channel_mask(recorded_chs)

        np.save(output_npy, tensor)
        with open(output_meta, "w") as f:
            json.dump({
                "subject_id": subject_id,
                "strategy": strategy,
                "num_channels": 64,
                "channel_names": CBRMOD_STANDARD_64,
                "active_channel_mask": active_channel_mask,
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

    files = find_eeg_files(src_dir)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Slicing Pipeline with Debug Logging for Interpolation & Window Quality Screening"
    )
    parser.add_argument("--src-dir", type=str, required=True, help="Input directory containing resampled files")
    parser.add_argument("--dst-dir", type=str, required=True, help="Output destination directory")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject id")
    parser.add_argument("--strategy", type=str, choices=["macro", "micro"], default="macro", help="Slicing strategy")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Window duration in seconds")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="CPU worker count")
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
