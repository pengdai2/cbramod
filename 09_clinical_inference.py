import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Import architecture from benchmark module
from 07_real_world_benchmark import CBraModRealWorldBenchmark


def top_k_percentile_pooling(
    probs: np.ndarray, 
    top_percentile: float = 0.10
) -> Union[float, np.ndarray]:
    """
    Aggregates window-level probabilities into subject-level class scores.
    Extracts the top-K highest predicted class probabilities for each class
    across windows and computes their mean.
    
    Args:
        probs: Array of window probabilities. 
               Shape: [Num_Windows] (1D) or [Num_Windows, Num_Classes] (2D)
        top_percentile: Top fraction of windows to average (default: 0.10 = top 10%)
        
    Returns:
        float if input is 1D, or np.ndarray [Num_Classes] if input is 2D.
    """
    if len(probs) == 0:
        return 0.0 if probs.ndim == 1 else np.array([])
    
    k = max(1, int(np.ceil(len(probs) * top_percentile)))
    
    if probs.ndim == 1:
        # 1D array: single target class (binary positive class)
        top_probs = np.sort(probs)[::-1][:k]
        return float(np.mean(top_probs))
    else:
        # 2D array: [Num_Windows, Num_Classes]
        # Sort along window axis (axis 0) for each class column independently
        sorted_probs = np.sort(probs, axis=0)[::-1, :]
        top_k_probs = sorted_probs[:k, :]
        return np.mean(top_k_probs, axis=0)  # Shape: [Num_Classes]


def run_subject_inference(
    model: torch.nn.Module,
    npy_path: Path,
    meta_path: Path,
    device: torch.device,
    filter_stage: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model inference across all window slices for a single subject.
    Optionally filters by sleep stage (e.g., 'N2').
    """
    # 1. Load window arrays and metadata sidecar
    window_data = np.load(npy_path)  # Shape: [Num_Windows, Channels, Time_Samples]
    
    stage_mask = None
    if filter_stage and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            stages = meta.get("stages", [])
            if stages:
                stage_mask = np.array([s == filter_stage for s in stages])

    # Apply stage filtering if requested and available
    if stage_mask is not None and len(stage_mask) == len(window_data):
        window_data = window_data[stage_mask]
        
    if len(window_data) == 0:
        return np.array([]), np.array([])

    # 2. Batch Inference
    model.eval()
    batch_size = 32
    window_probs = []

    with torch.no_grad():
        for i in range(0, len(window_data), batch_size):
            batch_np = window_data[i : i + batch_size].astype(np.float32)
            x_batch = torch.from_numpy(batch_np).to(device)
            
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            window_probs.append(probs.cpu().numpy())

    if not window_probs:
        return np.array([]), np.array([])

    all_window_probs = np.concatenate(window_probs, axis=0)  # Shape: [Num_Windows, Num_Classes]
    return all_window_probs, window_data


def evaluate_clinical_cohort(
    checkpoint_path: Path,
    test_manifest_path: Path,
    data_dir: Optional[Path] = None,
    num_channels: int = 64,
    num_classes: int = 2,
    filter_stage: str = "N2",
    top_percentile: float = 0.10,
    device_str: str = "cuda"
):
    """
    Executes full test set inference using top-10% percentile aggregation 
    to generate clinical subject-level diagnostics (Supports Binary and Multi-Class).
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"=== Running Clinical Inference Pipeline ({num_classes}-Class) on [{device}] ===")
    if data_dir:
        print(f"Data Root Directory: {data_dir}")

    # 1. Load Test Manifest
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    print(f"Loaded test set manifest with {len(test_df)} subject recordings.")

    # 2. Instantiate Model and Load Fine-Tuned Weights
    model = CBraModRealWorldBenchmark(num_channels=num_channels, num_classes=num_classes).to(device)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint state dict not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded fine-tuned checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    subject_results = []
    
    # 3. Patient-Level Inference Loop
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Subjects"):
        raw_npy_path = Path(row["npy_path"])
        ground_truth_label = int(row["label"])

        # Resolve relative npy_path against data_dir if provided
        if data_dir and not raw_npy_path.is_absolute():
            npy_path = data_dir / raw_npy_path
        else:
            npy_path = raw_npy_path

        subject_id = row.get("subject_id", raw_npy_path.stem)

        # Resolve relative meta_path against data_dir if provided
        if "meta_path" in row and pd.notna(row["meta_path"]):
            raw_meta_path = Path(row["meta_path"])
            if data_dir and not raw_meta_path.is_absolute():
                meta_path = data_dir / raw_meta_path
            else:
                meta_path = raw_meta_path
        else:
            meta_path = npy_path.with_suffix(".json")

        if not npy_path.exists():
            print(f"[Warning] Subject {subject_id} missing tensor file: {npy_path}, skipping...")
            continue

        if ground_truth_label == -1:
            print(f"[Warning] Subject {subject_id} missing label, skipping...")
            continue

        # Run window-level forward pass
        probs, _ = run_subject_inference(
            model=model,
            npy_path=npy_path,
            meta_path=meta_path,
            device=device,
            filter_stage=filter_stage
        )

        if len(probs) == 0:
            print(f"Warning: No valid windows evaluated for subject {subject_id}")
            continue

        # Aggregate via Top-10% Percentile Pooling
        if num_classes == 2:
            pos_probs = probs[:, 1]  # Extract Class 1 (Positive/Abnormal) probabilities
            patient_score = top_k_percentile_pooling(pos_probs, top_percentile=top_percentile)
            predicted_class = 1 if patient_score >= 0.5 else 0

            subject_results.append({
                "subject_id": subject_id,
                "ground_truth": ground_truth_label,
                "patient_score": patient_score,
                "predicted_class": predicted_class,
                "total_windows_evaluated": len(probs)
            })
        else:
            class_scores = top_k_percentile_pooling(probs, top_percentile=top_percentile)  # Shape: [Num_Classes]
            predicted_class = int(np.argmax(class_scores))

            res_entry = {
                "subject_id": subject_id,
                "ground_truth": ground_truth_label,
                "predicted_class": predicted_class,
                "total_windows_evaluated": len(probs)
            }
            # Log per-class pooled scores
            for c in range(num_classes):
                res_entry[f"prob_class_{c}"] = float(class_scores[c])
            
            subject_results.append(res_entry)

    # 4. Generate Cohort Diagnostic Performance Metrics
    results_df = pd.DataFrame(subject_results)
    
    y_true = results_df["ground_truth"].values
    y_pred = results_df["predicted_class"].values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 65)
    print("=== PATIENT-LEVEL CLINICAL INFERENCE REPORT ===")
    print("=" * 65)
    print(f"Total Test Subjects Evaluated:  {len(results_df)}")
    print(f"Number of Target Classes:       {num_classes}")
    print(f"Pooling Method:                 Top-{(top_percentile * 100):.0f}% Percentile Mean")
    print(f"Stage Filtering Applied:        {filter_stage if filter_stage else 'None (All Windows)'}")
    print("-" * 65)
    print(f"Accuracy:                       {acc * 100:.2f}%")
    print(f"Macro F1-Score:                 {macro_f1:.4f}")

    # Binary vs. Multi-class specific metric reports
    if num_classes == 2:
        y_scores = results_df["patient_score"].values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        print(f"ROC-AUC Score:                  {roc_auc:.4f}")
        print(f"Sensitivity (Recall):           {sensitivity * 100:.2f}% ({tp}/{tp + fn})")
        print(f"Specificity:                    {specificity * 100:.2f}% ({tn}/{tn + fp})")
    else:
        # Extract [N_subjects, N_classes] matrix for multi-class ROC-AUC
        prob_cols = [f"prob_class_{c}" for c in range(num_classes)]
        y_score_matrix = results_df[prob_cols].values

        try:
            roc_auc = roc_auc_score(y_true, y_score_matrix, multi_class="ovr", average="macro")
            print(f"Multi-Class ROC-AUC (OvR):      {roc_auc:.4f}")
        except Exception as e:
            print(f"Multi-Class ROC-AUC (OvR):      N/A ({e})")

        print("-" * 65)
        print("Detailed Classification Report:\n")
        print(classification_report(y_true, y_pred, digits=4))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    print("=" * 65)

    # Save patient-level results CSV
    output_csv = test_manifest_path.parent / "patient_level_test_predictions.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Detailed subject predictions saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-Level Clinical Inference & Top-10% Pooling")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best fine-tuned model state dict (.pt)")
    parser.add_argument("--test_manifest", type=str, required=True, help="Path to test_manifest.csv")
    parser.add_argument("--data_dir", type=str, default=None, help="Top-level root directory where relative tensor/meta files reside")
    parser.add_argument("--num_channels", type=int, default=64, help="EEG Channels count")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of target classes")
    parser.add_argument("--filter_stage", type=str, default=None, help="Optional sleep stage filter (e.g. N2)")
    parser.add_argument("--top_percentile", type=float, default=0.10, help="Top percentile pooling ratio (default 0.10)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    evaluate_clinical_cohort(
        checkpoint_path=Path(args.checkpoint),
        test_manifest_path=Path(args.test_manifest),
        data_dir=data_dir,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        filter_stage=args.filter_stage,
        top_percentile=args.top_percentile
    )
