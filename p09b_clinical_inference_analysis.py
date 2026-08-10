"""
Patient-Level Clinical Inference & Bootstrapping Analysis Script.
Reads CSV score reports, computes 95% bootstrapped CIs, and plots score distributions.

Usage:
    python analyze_test_scores.py \
        --input-csv test_scores.csv \
        --threshold 0.5600 \
        --n-bootstraps 2000 \
        --output-plot distribution.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze patient-level classification scores.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to input test scores CSV")
    parser.add_argument("--threshold", type=float, default=0.5600, help="Operating classification threshold")
    parser.add_argument("--n-bootstraps", type=int, default=2000, help="Number of bootstrap iterations for CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-plot", type=str, default="subject_score_distribution.png", help="Output path for plot")
    return parser.parse_args()


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Loads CSV score report and validates expected schema."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Score file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"subject_id", "ground_truth", "pooled_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["ground_truth"] = df["ground_truth"].astype(int)
    df["pooled_score"] = df["pooled_score"].astype(float)
    return df


def compute_bootstrapped_ci(
    df: pd.DataFrame,
    threshold: float = 0.5600,
    n_bootstraps: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Computes non-parametric bootstrapped 95% Confidence Intervals for clinical metrics.
    """
    y_true = df["ground_truth"].values
    y_scores = df["pooled_score"].values

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    bootstrapped_metrics = {
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "Macro F1": [],
        "ROC-AUC": []
    }

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_scores = y_scores[indices]

        # Skip iteration if bootstrap sample lacks representation for both classes
        if len(np.unique(boot_y_true)) < 2:
            continue

        boot_y_pred = (boot_y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(boot_y_true, boot_y_pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        macro_f1 = f1_score(boot_y_true, boot_y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(boot_y_true, boot_y_scores)

        bootstrapped_metrics["Accuracy"].append(acc)
        bootstrapped_metrics["Sensitivity"].append(sens)
        bootstrapped_metrics["Specificity"].append(spec)
        bootstrapped_metrics["Macro F1"].append(macro_f1)
        bootstrapped_metrics["ROC-AUC"].append(auc)

    results = []
    for metric, values in bootstrapped_metrics.items():
        point_est = np.mean(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        results.append({
            "Metric": metric,
            "Mean Estimate": point_est,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "Formatted CI": f"{point_est:.4f} [{ci_lower:.4f} - {ci_upper:.4f}]"
        })

    return pd.DataFrame(results)


def plot_subject_score_distribution(
    df: pd.DataFrame,
    threshold: float = 0.5600,
    save_path: Path = Path("subject_score_distribution.png")
):
    """
    Plots jittered strip distribution of subject scores, highlighting false positives 
    with their corresponding subject_id labels.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    df = df.copy()
    df["predicted_class"] = (df["pooled_score"] >= threshold).astype(int)

    # Classify prediction status
    status = []
    for _, row in df.iterrows():
        true = row["ground_truth"]
        pred = row["predicted_class"]
        if true == 1 and pred == 1:
            status.append("True Positive")
        elif true == 0 and pred == 0:
            status.append("True Negative")
        elif true == 0 and pred == 1:
            status.append("False Positive")
        else:
            status.append("False Negative")

    df["Classification"] = status
    df["Ground Truth Label"] = df["ground_truth"].map({1: "Patient (1)", 0: "Control (0)"})

    palette = {
        "True Positive": "#2ca02c",
        "True Negative": "#1f77b4",
        "False Positive": "#d62728",
        "False Negative": "#ff7f0e"
    }

    # Plot jittered strip distribution
    sns.stripplot(
        data=df,
        x="Ground Truth Label",
        y="pooled_score",
        hue="Classification",
        palette=palette,
        jitter=0.2,
        size=9,
        alpha=0.85,
        ax=ax
    )

    # Add operating threshold reference line
    ax.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Decision Threshold (T = {threshold:.4f})"
    )

    # Highlight and label misclassifications (False Positives & False Negatives) directly on plot
    misclassified = df[df["Classification"].isin(["False Positive", "False Negative"])]
    for _, row in misclassified.iterrows():
        x_pos = 1 if row["ground_truth"] == 1 else 0
        ax.annotate(
            row["subject_id"],
            xy=(x_pos, row["pooled_score"]),
            xytext=(x_pos + 0.12, row["pooled_score"]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
            fontsize=9,
            fontweight="bold",
            color="#d62728" if row["Classification"] == "False Positive" else "#ff7f0e"
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Patient-Level Probability Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Subject-Level Score P(Y=1)", fontsize=12)
    ax.set_xlabel("Clinical Ground Truth", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot successfully saved to: {save_path}")


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    output_plot_path = Path(args.output_plot)

    print(f"Loading patient scores from: {csv_path}")
    df = load_and_validate_csv(csv_path)
    print(f"Loaded {len(df)} subject evaluations.")

    # 1. Compute Bootstrapped CIs
    ci_df = compute_bootstrapped_ci(
        df,
        threshold=args.threshold,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed
    )

    print("\n=========================================================")
    print(f"   BOOTSTRAPPED 95% CONFIDENCE INTERVALS (B={args.n_bootstraps})   ")
    print("=========================================================")
    print(ci_df[["Metric", "Formatted CI"]].to_string(index=False))
    print("=========================================================\n")

    # 2. Render Score Distribution Plot
    plot_subject_score_distribution(df, threshold=args.threshold, save_path=output_plot_path)


if __name__ == "__main__":
    main()
