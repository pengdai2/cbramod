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
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from cbramod_common import CBraModE2EClassifier, CachedFeatureSubjectDataset, LinearProbeHead, MLPProbeHead, PANSubjectEEGDataset, compute_pooled_scores, setup_common_cli_parser


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


@torch.no_grad()
def infer_subject_windows(
    model: torch.nn.Module, 
    x_tensor: torch.Tensor, 
    batch_size: int, 
    device: torch.device
) -> np.ndarray:
    """Runs batched inference over a subject's windows (raw EEG or cached features)."""
    window_probs = []
    num_windows = x_tensor.shape[0]

    for j in range(0, num_windows, batch_size):
        x_batch = x_tensor[j : j + batch_size].to(device)
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        window_probs.append(probs)

    return np.concatenate(window_probs, axis=0) if window_probs else np.array([])


def generate_subject_predictions(
    args: argparse.Namespace, model:
    torch.nn.Module,
    device: torch.device
):
    """
    Unified generator streaming (subject_id, ground_truth_label, window_probs)
    for both cached features (.pt) and raw EEG manifest datasets.
    """
    if args.test_features_pt:
        dataset = CachedFeatureSubjectDataset(args.test_features_pt)
        print(f"Loaded cached features for {len(dataset)} subjects.")

        for i in tqdm(range(len(dataset)), desc="Processing Subjects (Cached)"):
            subject_id, subj_feats, label = dataset[i]
            probs = infer_subject_windows(model, subj_feats, args.batch_size, device)
            yield subject_id, label, probs

    else:
        dataset = PANSubjectEEGDataset(
            manifest_csv=args.test_manifest,
            data_dir=args.data_dir,
            filter_stage=args.filter_stage,
            memory_map=True
        )
        print(f"Loaded raw EEG recording dataset for {len(dataset)} subjects.")

        for i in tqdm(range(len(dataset)), desc="Processing Subjects (Raw EEG)"):
            x_tensor, y_tensor, subject_id = dataset[i]
            label = int(y_tensor.item())
            probs = infer_subject_windows(model, x_tensor, args.batch_size, device)
            yield subject_id, label, probs


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


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    metric: str = "macro_f1"
) -> Tuple[float, float]:
    """
    Sweeps decision threshold values from 0.01 to 0.99 to find the threshold 
    that maximizes the specified subject-level performance metric.
    """
    best_t = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.01, 0.99, 99)

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        elif metric == "balanced_accuracy":
            sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
            spec = recall_score(y_true, preds, pos_label=0, zero_division=0)
            score = (sens + spec) / 2.0
        else:
            score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t), float(best_score)


def evaluate_clinical_cohort(
    args: argparse.Namespace
) -> None:
    """Executes full test set inference and clinical cohort evaluation."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"=== Running Clinical Inference Pipeline ({args.num_classes}-Class) on [{device}] ===")

    # 1. Instantiate the appropriate Model Architecture
    if args.test_features_pt:
        print("Instantiating isolated Probe Head for cached feature inference.")
        if args.head_type == "linear":
            model = LinearProbeHead(num_patches=args.num_patches,
                                    emb_dim=args.cbra_dim,
                                    num_classes=args.num_classes)
        else:
            model = MLPProbeHead(
                num_patches=args.num_patches, 
                emb_dim=args.cbra_dim, 
                hidden_dim=args.head_dim, 
                num_classes=args.num_classes, 
                dropout=args.dropout
            )
    else:
        print("Instantiating full CBraModE2EClassifier for raw waveform inference.")
        model = CBraModE2EClassifier(
            num_channels=args.num_channels,
            sfreq=args.sfreq,
            num_patches=args.num_patches,
            emb_dim=args.cbra_dim,
            hidden_dim=args.head_dim,
            num_classes=args.num_classes,
            head_type=args.head_type
        )

    # 2. Load Model Checkpoint (Head-Only or Full-Model)
    model, ckpt_thresholds, epoch = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    # 3. Determine Active Strategies
    if args.pooling_strategy == "all":
        active_strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]
    else:
        active_strategies = [args.pooling_strategy]

    subject_results = {strat: [] for strat in active_strategies}
    ground_truths = []
    subject_ids = []

    # 4. Stream Dataset Predictions & Apply Pooling
    for subject_id, ground_truth_label, window_probs in generate_subject_predictions(args, model, device):
        if len(window_probs) == 0:
            continue

        ground_truths.append(ground_truth_label)
        subject_ids.append(subject_id)

        for strat in active_strategies:
            if args.num_classes == 2:
                # Extract positive class probability array [N] for binary score aggregation
                pos_probs = window_probs[:, 1]
                score = compute_pooled_scores(
                    pos_probs, method=strat, top_percentile=args.top_percentile, t_window=args.t_window
                )
            else:
                score = compute_pooled_scores(
                    window_probs, method=strat, top_percentile=args.top_percentile, t_window=args.t_window
                )

            subject_results[strat].append(score)

    # 5. Cohort Metrics, Threshold Sweep, Confusion Matrix, and Exporting
    ground_truths = np.array(ground_truths)
    output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    operating_thresholds = {
        strat: get_operating_threshold(
            pooling_strategy=strat,
            override_threshold=args.override_threshold,
            ckpt_thresholds=ckpt_thresholds)
        for strat in active_strategies}
    
    analyze_subject_results(
        subject_results=subject_results,
        ground_truths=ground_truths,
        subject_ids=subject_ids,
        num_classes=args.num_classes,
        operating_thresholds=operating_thresholds,
        output_dir=output_dir
    )


def analyze_subject_results(
    subject_results: Dict[str, List[Union[float, np.ndarray]]],
    ground_truths: np.ndarray,
    subject_ids: List[str],
    num_classes: int = 2,
    operating_thresholds: Dict[str, float] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """
    Detailed clinical cohort analytics across all evaluated pooling strategies.
    Computes confusion matrices, sensitivity, specificity, ROC-AUC, checkpoint vs.
    test-optimal thresholds, and exports per-subject predictions to CSV.
    """
    analysis_summary = {}

    print("\n" + "=" * 80)
    print(f"  CLINICAL COHORT EVALUATION REPORT ({num_classes}-CLASS)")
    print("=" * 80)

    for strat, scores in subject_results.items():
        scores_arr = np.array(scores)

        if num_classes == 2:
            # 1. Retrieve checkpoint threshold or default to 0.5
            eval_t = operating_thresholds.get(strat, 0.5) if operating_thresholds else 0.5

            # 2. Compute empirical optimal threshold on test cohort for upper-bound ceiling comparison
            opt_t, opt_f1 = find_optimal_threshold(ground_truths, scores_arr, metric="macro_f1")

            # 3. Primary metric evaluation using checkpoint decision boundary
            eval_preds = (scores_arr >= eval_t).astype(int)
            acc = accuracy_score(ground_truths, eval_preds)
            macro_f1 = f1_score(ground_truths, eval_preds, average="macro", zero_division=0)
            sens = recall_score(ground_truths, eval_preds, pos_label=1, zero_division=0)
            spec = recall_score(ground_truths, eval_preds, pos_label=0, zero_division=0)

            try:
                auc = roc_auc_score(ground_truths, scores_arr) if len(np.unique(ground_truths)) > 1 else 0.5
            except Exception:
                auc = 0.5

            # Confusion matrix parameters
            cm = confusion_matrix(ground_truths, eval_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            analysis_summary[strat] = {
                "eval_threshold": eval_t,
                "optimal_threshold": opt_t,
                "optimal_macro_f1": opt_f1,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "sensitivity": sens,
                "specificity": spec,
                "roc_auc": auc,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }

            print(f"\n Strategy: [{strat.upper()}]")
            print(f"  ├─ Checkpoint Threshold:         {eval_t:.4f}")
            print(f"  ├─ Test-Optimal Threshold:       {opt_t:.4f} (Upper Bound F1: {opt_f1:.4f})")
            print(f"  ├─ Subject Accuracy:             {acc:.4f} ({tn + tp}/{len(ground_truths)})")
            print(f"  ├─ Subject Macro F1:             {macro_f1:.4f}")
            print(f"  ├─ Sensitivity (Recall):         {sens:.4f} ({tp}/{tp + fn})")
            print(f"  ├─ Specificity:                  {spec:.4f} ({tn}/{tn + fp})")
            print(f"  ├─ ROC-AUC:                      {auc:.4f}")
            print(f"  └─ Confusion Matrix:             TP={tp}, FP={fp}, TN={tn}, FN={fn}")

            export_scores = scores_arr
            export_preds = eval_preds

        else:
            # Multi-class evaluation via argmax
            eval_preds = np.argmax(scores_arr, axis=1)
            acc = accuracy_score(ground_truths, eval_preds)
            macro_f1 = f1_score(ground_truths, eval_preds, average="macro", zero_division=0)

            analysis_summary[strat] = {
                "accuracy": acc,
                "macro_f1": macro_f1
            }

            print(f"\n Strategy: [{strat.upper()}]")
            print(f"  ├─ Subject Accuracy:  {acc:.4f}")
            print(f"  └─ Subject Macro F1:  {macro_f1:.4f}")

            export_scores = [list(s) for s in scores_arr]
            export_preds = eval_preds

        # 4. Save CSV predictions if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"subject_predictions_{strat}.csv"
            df_out = pd.DataFrame({
                "subject_id": subject_ids,
                "ground_truth": ground_truths,
                "pooled_score": export_scores,
                "prediction": export_preds
            })
            df_out.to_csv(csv_path, index=False)
            print(f"  └─ Exported Subject Predictions -> {csv_path}")

    print("=" * 80 + "\n")
    return analysis_summary


def parse_cli_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Class Patient-Level Clinical Inference")

    setup_common_cli_parser(parser)

    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--test-manifest", type=str, help="Path to test_manifest.csv for raw .npy inference")
    test_group.add_argument("--test-features-pt", type=str, help="Path to pre-extracted test features (.pt)")

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
    misc_group.add_argument("--output-dir", type=str, default=None, help="Output directory for the subject analysis")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli_args()
    seed_everything(args.seed)

    evaluate_clinical_cohort(args)
