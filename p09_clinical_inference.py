import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cbramod_utils import seed_everything
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
from cbramod_common import CBraModE2EClassifier, compute_pooled_scores, setup_common_cli_parser


def evaluate_all_pooling_strategies(
    window_probs: np.ndarray,
    top_percentile: float = 0.10,
    t_window: float = 0.60
) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluates all supported pooling strategies on a subject's window probabilities."""
    strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]
    return {
        strat: compute_pooled_scores(
            window_probs, method=strat, top_percentile=top_percentile, t_window=t_window
        )
        for strat in strategies
    }


def load_model_checkpoint(
    model: torch.nn.Module, 
    checkpoint_path: Path, 
    device: torch.device
) -> Tuple[torch.nn.Module, dict, Union[int, str]]:
    """
    Loads checkpoint weights into the model architecture.
    Handles both head-only (backbone frozen / LP-FT) state dicts and full model state dicts.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint state dict not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract state dict dict structure if wrapped inside checkpoint metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    optimal_thresholds = checkpoint.get("optimal_thresholds", {}) if isinstance(checkpoint, dict) else {}
    epoch = checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"

    # Strategy 1: Attempt direct full-model state dict load (Full Fine-Tuning)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded full model checkpoint (strict=True) from epoch {epoch}.")
        return model, optimal_thresholds, epoch
    except Exception:
        pass

    # Strategy 2: Attempt head-only state dict load into model.head (Linear Probe / Head Frozen)
    head_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone.") and not k.startswith("encoder."):
            head_state_dict[k] = v

    if hasattr(model, "head") and head_state_dict:
        try:
            model.head.load_state_dict(head_state_dict, strict=True)
            print(f"Successfully loaded head-only state dict into model.head from epoch {epoch}.")
            return model, optimal_thresholds, epoch
        except Exception:
            pass

    # Strategy 3: Fallback load with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with strict=False from epoch {epoch}.")
    if missing_keys:
        print(f"  [Info] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  [Info] Unexpected keys: {len(unexpected_keys)}")

    return model, optimal_thresholds, epoch


def run_subject_inference(
    model: torch.nn.Module,
    npy_path: Path,
    meta_path: Path,
    device: torch.device,
    batch_size: int = 32,
    filter_stage: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Runs model inference across all window slices for a single subject.
    Supports single or comma-separated stage filters (e.g., 'N2,N3').
    """
    window_data = np.load(npy_path)  # Shape: [Num_Windows, Channels, Time_Samples]
    num_windows = len(window_data)
    
    stage_mask = None
    if filter_stage and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            stages = meta.get("stages", [])
            if stages:
                target_stages = [s.strip() for s in filter_stage.split(",")]
                stage_mask = np.array([s in target_stages for s in stages])

    if stage_mask is not None and len(stage_mask) == len(window_data):
        window_data = window_data[stage_mask]
        
    if len(window_data) == 0:
        return np.array([]), np.array([])

    model.eval()
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
    return all_window_probs, window_data, num_windows


def get_operating_threshold(
    pooling_strategy: str,
    override_threshold: Optional[float],
    ckpt_thresholds: Dict[str, float]
) -> float:
    """Determines the operating threshold based on the pooling strategy and override settings."""
    # Determine Operating Decision Threshold
    if override_threshold is not None:
        operating_threshold = override_threshold
    elif pooling_strategy in ckpt_thresholds:
        operating_threshold = ckpt_thresholds.get(pooling_strategy)
    else:
        operating_threshold = 0.5
    return operating_threshold


def evaluate_clinical_cohort(
    args: argparse.Namespace
) -> None:
    """Executes full test set inference and clinical cohort evaluation."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"=== Running Clinical Inference Pipeline ({args.num_classes}-Class) on [{device}] ===")

    # 1. Load Test Manifest
    test_manifest_path = Path(args.test_manifest)
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    print(f"Loaded test set manifest with {len(test_df)} subject recordings.")

    # 2. Instantiate Model Architecture
    model = CBraModE2EClassifier(
        num_channels=args.num_channels,
        sfreq=args.sfreq,
        num_patches=args.num_patches,
        emb_dim=args.cbra_dim,
        hidden_dim=args.head_dim,
        num_classes=args.num_classes,
        head_type=args.head_type
    )

    # 3. Load Model Checkpoint (Head-Only or Full-Model)
    model, ckpt_thresholds, epoch = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    # Determine Active Strategies
    if args.pooling_strategy == "all":
        active_strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]
    else:
        active_strategies = [args.pooling_strategy]

    subject_results = {strat: [] for strat in active_strategies}

    # 4. Patient-Level Inference Loop
    data_dir = Path(args.data_dir) if args.data_dir else None
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Subjects"):
        raw_npy_path = Path(row["npy_path"])
        ground_truth_label = int(row["label"])

        npy_path = data_dir / raw_npy_path if (data_dir and not raw_npy_path.is_absolute()) else raw_npy_path
        subject_id = row.get("subject_id", raw_npy_path.stem)

        if "meta_path" in row and pd.notna(row["meta_path"]):
            raw_meta_path = Path(row["meta_path"])
            meta_path = data_dir / raw_meta_path if (data_dir and not raw_meta_path.is_absolute()) else raw_meta_path
        else:
            meta_path = npy_path.with_suffix(".json")

        if not npy_path.exists() or ground_truth_label == -1:
            continue

        probs, _, num_windows = run_subject_inference(
            model=model,
            npy_path=npy_path,
            meta_path=meta_path,
            device=device,
            batch_size=args.batch_size,
            filter_stage=args.filter_stage
        )

        if len(probs) == 0:
            continue

        # Extract pooled scores for active strategies
        if args.num_classes == 2:
            pos_probs = probs[:, 1]  # Class 1 probabilities
            if args.pooling_strategy == "all":
                pooled_dict = evaluate_all_pooling_strategies(pos_probs, top_percentile=args.top_percentile, t_window=args.t_window)
            else:
                pooled_dict = {
                    args.pooling_strategy: compute_pooled_scores(
                        pos_probs, method=args.pooling_strategy, top_percentile=args.top_percentile, t_window=args.t_window
                    )
                }

            for strat in active_strategies:
                score = pooled_dict[strat]
                operating_threshold = get_operating_threshold(
                    pooling_strategy=strat,
                    override_threshold=args.override_threshold,
                    ckpt_thresholds=ckpt_thresholds
                )
                pred_class = 1 if score >= operating_threshold else 0
                subject_results[strat].append({
                    "subject_id": subject_id,
                    "ground_truth": ground_truth_label,
                    "patient_score": score,
                    "predicted_class": pred_class,
                    "total_windows": num_windows,
                    "total_windows_evaluated": len(probs)
                })
        else:
            # Multi-Class Evaluation (> 2 classes)
            if args.pooling_strategy == "all":
                pooled_dict = evaluate_all_pooling_strategies(probs, top_percentile=args.top_percentile, t_window=args.t_window)
            else:
                pooled_dict = {
                    args.pooling_strategy: compute_pooled_scores(
                        probs, method=args.pooling_strategy, top_percentile=args.top_percentile, t_window=args.t_window
                    )
                }

            for strat in active_strategies:
                class_scores = pooled_dict[strat]  # Array of shape [num_classes]
                pred_class = int(np.argmax(class_scores))

                res_entry = {
                    "subject_id": subject_id,
                    "ground_truth": ground_truth_label,
                    "predicted_class": pred_class,
                    "total_windows": num_windows,
                    "total_windows_evaluated": len(probs)
                }
                for c in range(args.num_classes):
                    res_entry[f"prob_class_{c}"] = float(class_scores[c])

                subject_results[strat].append(res_entry)

    # 5. Analyze and Report Performance Metrics
    for strat in active_strategies:
        operating_threshold = get_operating_threshold(
            pooling_strategy=strat,
            override_threshold=args.override_threshold,
            ckpt_thresholds=ckpt_thresholds
        )
        analyze_subject_results(
            subject_results=subject_results[strat],
            strategy=strat,
            num_classes=args.num_classes,
            filter_stage=args.filter_stage,
            threshold=operating_threshold,
            test_manifest_path=test_manifest_path
        )


def analyze_subject_results(
    subject_results: List[Dict],
    strategy: str,
    num_classes: int,
    filter_stage: Optional[str],
    threshold: float,
    test_manifest_path: Path
):
    """Generates cohort diagnostic performance report and exports subject predictions."""
    if not subject_results:
        print(f"[Warning] No evaluation results generated for strategy: {strategy}")
        return

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
    print(f"Pooling Method:                 {strategy}")
    print(f"Stage Filtering Applied:        {filter_stage if filter_stage else 'None (All Windows)'}")
    print(f"Total Windows Evaluated:        {results_df['total_windows_evaluated'].sum()} / {results_df['total_windows'].sum()}")
    print(f"Operating Threshold Applied:    {threshold:.4f}")
    print("-" * 65)
    print(f"Accuracy:                       {acc * 100:.2f}%")
    print(f"Macro F1-Score:                 {macro_f1:.4f}")

    if num_classes == 2:
        y_scores = results_df["patient_score"].values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fpr, tpr_vals, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr_vals)

        print(f"ROC-AUC Score:                  {roc_auc:.4f}")
        print(f"Sensitivity (Recall):           {sensitivity * 100:.2f}% ({tp}/{tp + fn})")
        print(f"Specificity:                    {specificity * 100:.2f}% ({tn}/{tn + fp})")
    else:
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

    output_csv = test_manifest_path.parent / f"patient_level_test_predictions_{strategy}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Detailed subject predictions saved to: {output_csv}")


def parse_cli_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Class Patient-Level Clinical Inference")

    setup_common_cli_parser(parser)

    manifest_group = parser.add_argument_group("Test Manifest")
    manifest_group.add_argument("--test-manifest", type=str, required=True, help="Path to test_manifest.csv")

    ckpt_group = parser.add_argument_group("Model Checkpoint")
    ckpt_group.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # Pooling Strategy
    pool_group = parser.add_argument_group("Pooling Strategy")
    pool_group.add_argument(
        "--pooling-strategy", 
        type=str, 
        default="p85_score", 
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio", "all"],
        help="Pooling strategy choice (default: 'p85_score', or 'all' for full comparative report)"
    )
    pool_group.add_argument("--top-percentile", type=float, default=0.10, help="Top percentile ratio (default: 0.10)")
    pool_group.add_argument("--t-window", type=float, default=0.60, help="Window threshold for burden ratio (default: 0.60)")

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--override-threshold", type=float, default=None, help="Override operating decision threshold")
    misc_group.add_argument("--batch-size", type=int, default=512, help="Batch size for inference (default: 512)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli_args()
    seed_everything(args.seed)

    evaluate_clinical_cohort(args)
