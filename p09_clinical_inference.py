import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from cbramod_common import seed_everything
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from cbramod_common import (
    CachedFeatureSubjectDataset,
    PANSubjectEEGDataset,
    add_log_filename_argument,
    build_frozen_e2e_classifier,
    build_frozen_probe,
    compute_pooled_scores,
    extract_ckpt_metadata,
    find_optimal_threshold,
    get_operating_threshold,
    resolve_pooling_config,
    setup_inference_cli_parser,
)
from cbramod_utils import setup_logger


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
    if args.features_pt:
        dataset = CachedFeatureSubjectDataset(args.features_pt, filter_subject=args.subject_id)
        print(f"Loaded cached features for {len(dataset)} subjects.")

        for i in tqdm(range(len(dataset)), desc="Processing Subjects (Cached)"):
            subj_feats, label_tensor, subject_id, _, _ = dataset[i]
            label = int(label_tensor.item())
            probs = infer_subject_windows(model, subj_feats, args.batch_size, device)
            yield subject_id, label, probs

    else:
        dataset = PANSubjectEEGDataset(
            manifest_csv=args.manifest,
            data_dir=args.data_dir,
            filter_stage=args.filter_stage,
            filter_subject=args.subject_id,
            memory_map=True
        )
        print(f"Loaded raw EEG recording dataset for {len(dataset)} subjects.")

        for i in tqdm(range(len(dataset)), desc="Processing Subjects (Raw EEG)"):
            x_tensor, y_tensor, subject_id, _, _ = dataset[i]
            label = int(y_tensor.item())
            probs = infer_subject_windows(model, x_tensor, args.batch_size, device)
            yield subject_id, label, probs


def evaluate_clinical_cohort(
    args: argparse.Namespace, logger,
) -> None:
    """Executes full test set inference and clinical cohort evaluation."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"=== Running Clinical Inference Pipeline ({args.num_classes}-Class) on [{device}] ===")

    # 1. Instantiate the appropriate Model Architecture + load its checkpoint, metadata-first
    # (num_patches/cbra_dim/num_classes/num_channels/sfreq/head_type all resolved from the
    # checkpoint's own saved metadata when present, not blindly from CLI flags) and with the
    # checkpoint's own explicit checkpoint_kind (head_only vs. full_model) deciding how to load the
    # state dict -- replacing the old try/except-based load_model_checkpoint() guess.
    if args.features_pt:
        model, ckpt = build_frozen_probe(args, device, logger)
    else:
        model, ckpt = build_frozen_e2e_classifier(args, device, logger)
    ckpt_thresholds, _epoch, ckpt_pooling_params = extract_ckpt_metadata(ckpt)

    # 2b. Resolve Pooling Config: CLI override > checkpoint's training-time config > hardcoded default
    pooling_strategy, top_percentile, t_window = resolve_pooling_config(
        pooling_strategy=args.pooling_strategy,
        top_percentile=args.top_percentile,
        t_window=args.t_window,
        ckpt_pooling_params=ckpt_pooling_params
    )
    print(
        f"Pooling config: strategy={pooling_strategy}, top_percentile={top_percentile}, t_window={t_window} "
        f"(source: {'CLI override' if args.pooling_strategy is not None else 'checkpoint/default'})"
    )

    # 3. Determine Active Strategies
    if pooling_strategy == "all":
        active_strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]
    else:
        active_strategies = [pooling_strategy]

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
                    pos_probs, method=strat, top_percentile=top_percentile, t_window=t_window
                )
            else:
                score = compute_pooled_scores(
                    window_probs, method=strat, top_percentile=top_percentile, t_window=t_window
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
            per_class_recall = recall_score(
                ground_truths, eval_preds, average=None, zero_division=0,
                labels=np.arange(num_classes)
            )

            try:
                auc = roc_auc_score(
                    ground_truths, scores_arr, multi_class="ovr", average="macro",
                    labels=np.arange(num_classes)
                ) if len(np.unique(ground_truths)) > 1 else 0.5
            except Exception:
                auc = 0.5

            cm = confusion_matrix(ground_truths, eval_preds, labels=np.arange(num_classes))

            analysis_summary[strat] = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "per_class_recall": per_class_recall.tolist(),
                "roc_auc_ovr_macro": auc,
                "confusion_matrix": cm.tolist()
            }

            print(f"\n Strategy: [{strat.upper()}]")
            print(f"  ├─ Subject Accuracy:        {acc:.4f}")
            print(f"  ├─ Subject Macro F1:        {macro_f1:.4f}")
            print(f"  ├─ ROC-AUC (OVR macro):     {auc:.4f}")
            print(f"  ├─ Per-Class Recall:        {np.round(per_class_recall, 4).tolist()}")
            print(f"  └─ Confusion Matrix (rows=truth, cols=pred):\n{cm}")

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

            if num_classes == 2:
                # Bake in outcome (TP/TN/FP/FN) and confidence (distance from the
                # decision threshold that actually produced "prediction" above) so
                # downstream analysis (e.g. p09d_subject_confidence_report.py)
                # doesn't have to guess/re-supply eval_t and risk it not matching.
                is_correct = ground_truths == export_preds
                df_out["outcome"] = [
                    ("TP" if gt == 1 else "TN") if correct else ("FP" if pred == 1 else "FN")
                    for gt, pred, correct in zip(ground_truths, export_preds, is_correct)
                ]
                df_out["confidence"] = np.abs(scores_arr - eval_t)

            df_out.to_csv(csv_path, index=False)
            print(f"  └─ Exported Subject Predictions -> {csv_path}")

    print("=" * 80 + "\n")
    return analysis_summary


def parse_cli_args()-> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Multi-Class Patient-Level Clinical Inference")
    add_log_filename_argument(parser, __file__)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli_args()
    seed_everything(args.seed)
    logger = setup_logger(args.log_filename)

    evaluate_clinical_cohort(args, logger)
