import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

def parse_label_map(label_map_arg: str) -> Dict[str, int]:
    """Parses string-to-integer label map from JSON string or file path."""
    json_path = Path(label_map_arg)
    if json_path.is_file():
        with open(json_path, "r") as f:
            mapping = json.load(f)
    else:
        mapping = json.loads(label_map_arg)
    return {str(k): int(v) for k, v in mapping.items()}

def load_clinical_labels(
    labels_csv_path: Path, 
    subject_id_col: str = "subject_id", 
    label_col: str = "label",
    label_map: Optional[Dict[str, int]] = None
) -> Dict[str, Tuple[str, int]]:
    """
    Loads clinical labels CSV and maps subject IDs to a tuple of (raw_label, mapped_label).
    """
    df = pd.read_csv(labels_csv_path)
    
    if subject_id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Labels CSV must contain '{subject_id_col}' and '{label_col}' columns.")
        
    subject_map = {}
    for _, row in df.iterrows():
        subj_id = str(row[subject_id_col]).strip()
        raw_label = str(row[label_col]).strip()
        
        if label_map is not None:
            if raw_label in label_map:
                mapped_label = label_map[raw_label]
                subject_map[subj_id] = (raw_label, mapped_label)
        else:
            try:
                mapped_label = int(raw_label)
                subject_map[subj_id] = (raw_label, mapped_label)
            except ValueError:
                pass
            
    return subject_map

def create_subject_level_splits(
    subject_ids: List[str], 
    labels_dict: Optional[Dict[str, Tuple[str, int]]] = None,
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Splits subjects into Train/Val/Test splits at the subject level.
    Supports arbitrary Multi-Class integer labels using per-class stratified partitioning.
    """
    random.seed(seed)
    
    if labels_dict:
        # Group subject IDs by their integer mapped label (supports N classes: 0, 1, 2, ..., K-1)
        class_buckets: Dict[int, List[str]] = {}
        for s in subject_ids:
            lbl_tuple = labels_dict.get(s)
            lbl = lbl_tuple[1] if lbl_tuple else -1
            class_buckets.setdefault(lbl, []).append(s)
            
        train_subjs, val_subjs, test_subjs = [], [], []

        for class_lbl, subjs in class_buckets.items():
            random.shuffle(subjs)
            n = len(subjs)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_subjs.extend(subjs[:n_train])
            val_subjs.extend(subjs[n_train:n_train + n_val])
            test_subjs.extend(subjs[n_train + n_val:])

    else:
        # Fallback to simple random split if no labels map is supplied
        subjs = subject_ids.copy()
        random.shuffle(subjs)
        
        n = len(subjs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_subjs = subjs[:n_train]
        val_subjs = subjs[n_train:n_train + n_val]
        test_subjs = subjs[n_train + n_val:]

    return {
        "train": sorted(train_subjs),
        "val": sorted(val_subjs),
        "test": sorted(test_subjs)
    }

def build_manifest(
    sliced_dir: Path,
    labels_csv: Optional[Path],
    output_dir: Path,
    label_map: Optional[Dict[str, int]] = None,
    subject_id_col: str = "subject_id",
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """Gathers preprocessed `.npy` sliced files, joins multi-class labels, and exports manifest CSVs."""
    sliced_dir = Path(sliced_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_map = load_clinical_labels(labels_csv, subject_id_col, label_col, label_map) if labels_csv else None

    meta_files = sorted(sliced_dir.rglob("*_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.json files found in {sliced_dir}. Run 02_slice_eeg_dataset.py first.")

    subject_data_records = []
    found_subjects = set()

    for meta_path in meta_files:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        subj_id = meta["subject_id"]
        found_subjects.add(subj_id)
        
        npy_path = meta_path.with_name(f"{subj_id}_windows.npy")
        if not npy_path.exists():
            continue

        raw_label, mapped_label = labels_map.get(subj_id, ("unlabeled", -1)) if labels_map else ("unlabeled", -1)

        # Calculate relative paths for dataset portability
        rel_npy = npy_path.relative_to(sliced_dir)
        rel_meta = meta_path.relative_to(sliced_dir)

        subject_data_records.append({
            "subject_id": subj_id,
            "npy_path": str(rel_npy),
            "meta_path": str(rel_meta),
            "num_slices": meta["num_slices"],
            "sampling_freq": meta["sampling_freq"],
            "raw_label": raw_label,
            "label": mapped_label
        })

    # Subject-level non-overlapping multi-class stratified split
    split_map = create_subject_level_splits(
        subject_ids=list(found_subjects),
        labels_dict=labels_map,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    subj_to_split = {}
    for split_name, subjs in split_map.items():
        for s in subjs:
            subj_to_split[s] = split_name

    manifest_rows = []
    for rec in subject_data_records:
        rec["split"] = subj_to_split.get(rec["subject_id"], "unknown")
        manifest_rows.append(rec)

    full_df = pd.DataFrame(manifest_rows)

    # Export master CSV and split-specific CSVs
    full_df.to_csv(output_dir / "master_manifest.csv", index=False)
    
    for split in ["train", "val", "test"]:
        split_df = full_df[full_df["split"] == split]
        split_df.to_csv(output_dir / f"{split}_manifest.csv", index=False)

    print("=== Multi-Class Dataset Labeling & Stratified Split Summary ===")
    print(f"Total Unique Subjects: {len(found_subjects)}")
    if labels_map:
        unique_classes = sorted(list(set(l[1] for l in labels_map.values())))
        print(f"Detected Target Classes: {unique_classes}")
        
    for split in ["train", "val", "test"]:
        sub_df = full_df[full_df["split"] == split]
        total_windows = sub_df["num_slices"].sum()
        num_subjs = len(sub_df)
        
        # Multi-class distribution string
        if labels_map:
            class_counts = sub_df["label"].value_counts().to_dict()
            dist_str = ", ".join([f"Class {k}: {class_counts.get(k, 0)}" for k in sorted(class_counts.keys())])
        else:
            dist_str = "No Labels Provided"
            
        print(f" - {split.upper()} Set: {num_subjs} Subj | {total_windows} Slices | Distribution: [{dist_str}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Class Dataset Labeling and Stratified Split")
    parser.add_argument("--sliced_dir", type=str, required=True, help="Directory containing sliced .npy and _meta.json files")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to clinical labels CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for CSV manifests")
    parser.add_argument("--label_map", type=str, default=None, help="JSON string or path to JSON file for string-to-int mapping")
    parser.add_argument("--subject_id_col", type=str, default="subject_id", help="Column name for subject ID in labels CSV")
    parser.add_argument("--label_col", type=str, default="label", help="Column name for raw label in labels CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject split reproducibility")

    args = parser.parse_args()

    parsed_map = parse_label_map(args.label_map) if args.label_map else None

    build_manifest(
        sliced_dir=Path(args.sliced_dir),
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
        output_dir=Path(args.output_dir),
        label_map=parsed_map,
        subject_id_col=args.subject_id_col,
        label_col=args.label_col,
        seed=args.seed
    )
