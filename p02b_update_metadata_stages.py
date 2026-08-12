import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union
from cbramod_utils import extract_epoch_stages
from cbramod_utils import find_eeg_files, load_raw_eeg, valid_regex
from cbramod_utils import predict_epoch_stages_yasa
from tqdm import tqdm


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

    files = find_eeg_files(src_dir)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update existing dataset metadata JSON files with sleep stages")
    parser.add_argument("--src-dir", type=str, required=True, help="Input directory containing raw EEG/PSG files")
    parser.add_argument("--dst-dir", type=str, required=True, help="Directory containing existing _meta.json files")
    parser.add_argument("--pattern", type=valid_regex, default=None, help="Optional regex pattern to match subject ID")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Fallback window duration in seconds")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="CPU worker count")
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
