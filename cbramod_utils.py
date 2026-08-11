import argparse
import logging
from pathlib import Path

import re
import sys
from typing import List, Tuple

import mne
import yasa


def load_raw_eeg(file_path: Path) -> mne.io.BaseRaw:
    """Loads raw EEG file using the appropriate MNE reader with preload enabled."""
    ext = file_path.suffix.lower()
    if ext == ".edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif ext == ".fif":
        return mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    elif ext == ".vhdr":
        return mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    elif ext == ".bdf":
        return mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
    elif ext == ".set":
        return mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


SUPPORTED_EXTENSIONS = {".edf", ".fif", ".vhdr", ".bdf", ".set"}


def find_eeg_files(src_dir: Path) -> List[Path]:
    """Recursively gather all supported EEG files in the source directory."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))
    return sorted(files)


def valid_regex(pattern_string):
    try:
        return re.compile(pattern_string)
    except re.error:
        raise argparse.ArgumentTypeError(f"Invalid regex: '{pattern_string}'")


def setup_logger(log_path: Path) -> logging.Logger:
    """Configures structured logging to stdout and file."""
    logger = logging.getLogger("CBraModPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 🌟 STOP LOGS FROM BUBBLING UP TO THE ROOT LOGGER
    logger.propagate = False

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(log_path)
    f_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(f_handler)

    return logger


def evaluate_subject_quality(
    meta: dict,
    max_rejection_rate: float = 0.15,
    max_n3_rejection_rate: float = 0.10,
    min_valid_hours: float = 4.5,
    window_duration_sec: float = 30.0
) -> Tuple[bool, dict, str]:
    """
    Evaluates slice-level metadata for quality screening and subject acceptance.

    Checks:
    1. Overall window rejection rate <= max_rejection_rate
    2. N3 stage window rejection rate <= max_n3_rejection_rate
    3. Total usable sleep duration >= min_valid_hours
    """
    slices = meta.get("slices", [])
    stages = meta.get("stages", [])
    total_slices = meta.get("num_slices", len(slices))

    if total_slices == 0:
        return False, {}, "EMPTY_RECORDING"

    # Extract or infer slice validity
    if slices:
        valid_mask = [s.get("is_valid", True) for s in slices]
    else:
        # Fallback if slice details are omitted
        valid_mask = [True] * total_slices

    total_valid = sum(valid_mask)
    total_invalid = total_slices - total_valid
    overall_rejection_rate = total_invalid / float(total_slices)
    valid_hours = (total_valid * window_duration_sec) / 3600.0

    # Calculate N3 stage specific rejection rate if stage data is present
    n3_rejection_rate = 0.0
    if stages and len(stages) == len(valid_mask):
        n3_pairs = [(v, st) for v, st in zip(valid_mask, stages) if str(st).upper() == "N3"]
        if n3_pairs:
            n3_total = len(n3_pairs)
            n3_invalid = sum(1 for v, _ in n3_pairs if not v)
            n3_rejection_rate = n3_invalid / float(n3_total)

    # Evaluate Gatekeeping Rules
    reasons = []
    if overall_rejection_rate > max_rejection_rate:
        reasons.append(f"HIGH_REJECTION_RATE ({overall_rejection_rate:.1%})")
    if n3_rejection_rate > max_n3_rejection_rate:
        reasons.append(f"HIGH_N3_REJECTION ({n3_rejection_rate:.1%})")
    if valid_hours < min_valid_hours:
        reasons.append(f"INSUFFICIENT_SLEEP_DURATION ({valid_hours:.2f}h)")

    is_accepted = len(reasons) == 0
    reason_str = "PASSED" if is_accepted else "; ".join(reasons)

    metrics = {
        "total_slices": total_slices,
        "valid_slices": total_valid,
        "invalid_slices": total_invalid,
        "overall_rejection_rate": round(overall_rejection_rate, 4),
        "n3_rejection_rate": round(n3_rejection_rate, 4),
        "valid_sleep_hours": round(valid_hours, 2),
        "gatekeeping_status": "ACCEPTED" if is_accepted else "REJECTED",
        "rejection_reason": reason_str
    }

    return is_accepted, metrics, reason_str


# Standard sleep stage string normalization map
STAGE_NORM_MAP = {
    "sleep stage w": "W", "stage w": "W", "wake": "W", "0": "W",
    "sleep stage n1": "N1", "stage 1": "N1", "n1": "N1", "1": "N1",
    "sleep stage n2": "N2", "stage 2": "N2", "n2": "N2", "2": "N2",
    "sleep stage n3": "N3", "stage 3": "N3", "stage 4": "N3", "n3": "N3", "3": "N3", "4": "N3",
    "sleep stage r": "REM", "stage rem": "REM", "rem": "REM", "5": "REM"
}


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
