import argparse
import logging
from pathlib import Path

import random
import re
import sys
from typing import List

import numpy as np
import mne
import torch


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


def seed_everything(seed: int = 42) -> None:
    """Ensures end-to-end reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
