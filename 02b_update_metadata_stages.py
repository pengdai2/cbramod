import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union
import mne
import yasa
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
        return mne.io.read_raw_fif(file_path, preload=False, verbose=False)
    elif ext == ".edf":
        return mne.io.read_raw_edf(file_path, preload=False, verbose=False)
    elif ext == ".bdf":
        return mne.io.read_raw_bdf(file_path, preload=False, verbose=False)
    elif ext == ".vhdr":
        return mne.io.read_raw_brainvision(file_path, preload=False, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_epoch_stages(raw: mne.io.BaseRaw, num_windows: int, window_sec: float = 30.0) -> List[str]:
    """Parses MNE raw annotations and maps each window midpoint to its sleep stage."""
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


def process_metadata_update_worker(
    args_tuple: Tuple[Path, Path, re.Pattern, float, bool]
) -> Dict[str, Union[str, int]]:
    """Worker task to update a single subject's metadata JSON file."""
    src_file, dst_dir, pattern, fallback_window_sec, force = args_tuple

    subject_id = src_file.stem
    if pattern:
        match = pattern.search(subject_id)
        if match:
            subject_id = match.group(0)

    # Search for target metadata file in mirror subfolder or flat dst_dir
    meta_path = dst_dir / f"{subject_id}_meta.json"
    if not meta_path.exists():
        # Fallback check directly inside dst_dir
        meta_path = dst_dir.parent / f"{subject_id}_meta.json"

    if not meta_path.exists():
        return {"status": "MISSING_JSON", "subject": subject_id, "count": 0}

    try:
        # Load existing JSON
        with open(meta_path, "r") as f:
            meta_data = json.load(f)

        # Check if stages are already populated
        if "stages" in meta_data and len(meta_data["stages"]) > 0 and not force:
            return {"status": "SKIPPED", "subject": subject_id, "count": 0}

        num_slices = meta_data.get("num_slices", len(meta_data.get("slices", [])))
        if num_slices == 0:
            return {"status": "NO_SLICES", "subject": subject_id, "count": 0}

        # Determine window duration from slice metadata or fallback
        window_sec = fallback_window_sec
        if "slices" in meta_data and len(meta_data["slices"]) > 0:
            first_slice = meta_data["slices"][0]
            if "start_sec" in first_slice and "end_sec" in first_slice:
                window_sec = first_slice["end_sec"] - first_slice["start_sec"]

        # Load raw file and extract annotations
        raw = load_raw_eeg(src_file)

        # Check if raw has native annotations; if not, use YASA predictor
        if len(raw.annotations) > 0:
            # Native annotation extraction
            stages = extract_epoch_stages(raw, num_windows=num_slices, window_sec=window_sec)
        else:
            # Automated prediction via YASA
            stages = predict_epoch_stages_yasa(raw, num_windows=num_slices)

        # Inject stages key into top level of JSON
        meta_data["stages"] = stages

        # Save updated metadata in place
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

        return {"status": "SUCCESS", "subject": subject_id, "count": len(stages)}

    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "subject": subject_id, "count": 0}


def run_metadata_update_pipeline(
    src_dir: Path,
    dst_dir: Path,
    pattern: re.Pattern = None,
    window_sec: float = 30.0,
    num_workers: int = 1,
    force: bool = False
):
    """Executes parallel updating of metadata JSON files with sleep stages."""
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))
    files = sorted(files)

    print(f"Found {len(files)} raw files to match against metadata in: {dst_dir}")

    tasks = [
        (f, dst_dir / f.relative_to(src_dir).parent, pattern, window_sec, force)
        for f in files
    ]

    results = {"SUCCESS": 0, "SKIPPED": 0, "MISSING_JSON": 0, "ERROR": 0, "UPDATED_STAGES": 0}

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_metadata_update_worker, task) for task in tasks]
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Updating Metadata", unit="subj")
            for future in pbar:
                res = future.result()
                status = res["status"].split(":")[0]
                results[status] = results.get(status, 0) + 1
                results["UPDATED_STAGES"] += res["count"]
                pbar.set_postfix({"Status": status, "Stages": res["count"]})
    else:
        pbar = tqdm(tasks, desc="Updating Metadata (Single Process)", unit="subj")
        for task in pbar:
            res = process_metadata_update_worker(task)
            status = res["status"].split(":")[0]
            results[status] = results.get(status, 0) + 1
            results["UPDATED_STAGES"] += res["count"]
            pbar.set_postfix({"Status": status, "Stages": res["count"]})

    print("\n=== Metadata Stage Update Summary ===")
    print(f" - Updated Metadata Files: {results['SUCCESS']}")
    print(f" - Skipped (Already Had Stages): {results['SKIPPED']}")
    print(f" - Missing JSON Files:     {results['MISSING_JSON']}")
    print(f" - Errors:                 {results['ERROR']}")
    print(f" - Total Stage Labels Added: {results['UPDATED_STAGES']}")


def valid_regex(pattern_string):
    try:
        return re.compile(pattern_string)
    except re.error:
        raise argparse.ArgumentTypeError(f"Invalid regex: '{pattern_string}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update existing dataset metadata JSON files with sleep stages")
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing raw EEG/PSG files")
    parser.add_argument("--dst_dir", type=str, required=True, help="Directory containing existing _meta.json files")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject ID")
    parser.add_argument("--window_sec", type=float, default=30.0, help="Fallback window duration in seconds")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="CPU worker count")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing stage fields in JSON")

    args = parser.parse_args()

    run_metadata_update_pipeline(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        pattern=args.pattern,
        window_sec=args.window_sec,
        num_workers=args.num_workers,
        force=args.force
    )
