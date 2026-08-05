import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def run_performance_analysis(
    predictions_csv: Path,
    output_dir: Path,
    target_names: list = None
):
    """
    Computes confusion matrix, ROC-AUC, Precision-Recall curves, 
    and extracts error cases from patient prediction CSVs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")
        
    df = pd.read_csv(predictions_csv)
    print(f"Loaded predictions for {len(df)} subjects from: {predictions_csv.name}")

    y_true = df["ground_truth"].values
    y_scores = df["patient_score"].values
    y_pred = df["predicted_class"].values

    if target_names is None:
        target_names = ["Control / Normal", "Abnormal / Target"]

    # 1. Primary Diagnostic Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1_macro = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0

    # Calculate Specificity from Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 60)
    print("=== COHORT DIAGNOSTIC PERFORMANCE SUMMARY ===")
    print("=" * 60)
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Sensitivity (TPR): {rec * 100:.2f}% ({tp}/{tp + fn})")
    print(f"Specificity (TNR): {spec * 100:.2f}% ({tn}/{tn + fp})")
    print(f"Precision (PPV):   {prec * 100:.2f}% ({tp}/{tp + fp})")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print("=" * 60)

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names,
        cbar=False
    )
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Ground Truth Diagnosis")
    plt.title("Patient-Level Confusion Matrix")
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # 3. ROC & Precision-Recall Curves Plot
    if len(np.unique(y_true)) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        axes[0].set_xlabel("1 - Specificity (False Positive Rate)")
        axes[0].set_ylabel("Sensitivity (True Positive Rate)")
        axes[0].set_title("Receiver Operating Characteristic (ROC)")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Precision-Recall Curve
        precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_pts, precision_pts)
        axes[1].plot(recall_pts, precision_pts, color="purple", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
        axes[1].set_xlabel("Recall (Sensitivity)")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].legend(loc="lower left")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        curves_path = output_dir / "diagnostic_curves.png"
        plt.savefig(curves_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"ROC/PR curves saved to: {curves_path}")

    # 4. Clinical Error Analysis: Extract Misclassified Subjects
    false_positives = df[(df["ground_truth"] == 0) & (df["predicted_class"] == 1)]
    false_negatives = df[(df["ground_truth"] == 1) & (df["predicted_class"] == 0)]

    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"False Positives (Over-called): {len(false_positives)} subjects")
    print(f"False Negatives (Missed):     {len(false_negatives)} subjects")

    # Export detailed misclassification log
    error_df = pd.concat([false_positives, false_negatives], axis=0).sort_values("subject_id")
    error_csv = output_dir / "misclassified_subjects.csv"
    error_df.to_csv(error_csv, index=False)
    print(f"Misclassified subjects exported to: {error_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBraMod Performance Analysis & Error Extraction")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to patient_level_test_predictions.csv")
    parser.add_argument("--output_dir", type=str, default="./performance_analysis", help="Output directory")

    args = parser.parse_args()

    run_performance_analysis(
        predictions_csv=Path(args.predictions_csv),
        output_dir=Path(args.output_dir)
    )