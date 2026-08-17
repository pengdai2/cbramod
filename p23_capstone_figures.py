"""
p23_capstone_figures.py

Three summary figures capping the sigma/delta causal investigation (see
docs/sigma_band_causal_investigation.md, Chapters 10-11) -- built entirely from already-saved
p09k (or p09f) window-level CSVs, exactly like p22_ground_truth_band_power_comparison.py. No new
model inference needed; the "probability" column in those CSVs is model[0]'s own already-computed
window-level output, saved at the time p09k/p09f were originally run.

Each figure targets a specific claim this investigation makes, rather than defaulting to whichever
chart looks simplest -- see the module docstrings in p21/p22 and Chapter 10 of the writeup for the
full reasoning behind why these are the right three, and why a naive pooled-window histogram is
deliberately NOT one of them (it collapses the between-subject/within-subject distinction this
investigation has been careful to keep separate throughout).

  1. between_subject.png     -- per-subject means, patient vs control: box + jittered points.
                                 Answers "is there a broad, cohort-wide group difference?"
  2. within_subject_shape.png -- per-subject percentile shape (p10-p99), averaged within each group.
                                 Answers "is a typical subject's own shift broad, or tail-concentrated?"
  3. window_level_relationship.png -- window_prob vs. sigma/delta power, WITHIN-SUBJECT CENTERED
                                 (each window's value minus that subject's own mean) before binning,
                                 split by group. Answers the piece the other two don't: does
                                 window_prob actually TRACK real signal, including WITHIN each group
                                 separately (not just as a byproduct of patients/controls differing
                                 on both quantities independently, or of subject-to-subject baseline
                                 spread within a group swamping the true within-subject slope)?

Usage:
    python p23_capstone_figures.py \
        --window-csv val_ckpt/absolute_band_power_analysis.csv \
        --window-csv test_ckpt/absolute_band_power_analysis.csv \
        --output-dir figures/
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p22_ground_truth_band_power_comparison import PERCENTILES, build_subject_level_summary, spearman_corr

# Consistent styling across all three figures.
GROUP_STYLE = {
    1: {"label": "Patient", "color": "#d62728", "linestyle": "-", "marker": "o"},
    0: {"label": "Control", "color": "#1f77b4", "linestyle": "--", "marker": "s"},
}

# The five quantities every figure walks through, in a fixed order.
FEATURE_SPECS = [
    {"col": "probability", "label": "Model window probability"},
    {"col": "sigma_real_abspower", "label": "Sigma power (abs, µV²)"},
    {"col": "delta_real_abspower", "label": "Delta power (abs, µV²)"},
    {"col": "n_spindles", "label": "Spindle count (YASA)"},
    {"col": "n_slow_waves", "label": "Slow-wave count (YASA)"},
]

# The two bands used for the window-level relationship figure specifically.
RELATIONSHIP_BANDS = [
    {"col": "sigma_real_abspower", "label": "Sigma power (abs, µV²)"},
    {"col": "delta_real_abspower", "label": "Delta power (abs, µV²)"},
]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Builds the three capstone figures (between-subject, within-subject shape, "
                    "window-level relationship) from already-saved p09k/p09f window-level CSVs -- "
                    "no new model inference needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window-csv", action="append", required=True, dest="window_csvs",
        help="Path to a p09k (or p09f) output CSV. Repeat this flag to combine multiple runs."
    )
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument(
        "--n-bins", type=int, default=15,
        help="Number of quantile bins per group for the window-level relationship trend lines."
    )
    parser.add_argument(
        "--scatter-sample", type=int, default=400,
        help="Number of windows per group to show as a faint background scatter in Figure 3 "
             "(0 disables the scatter, showing only the binned trend line)."
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_windows(window_csvs: List[str]) -> pd.DataFrame:
    frames = []
    for csv_path in window_csvs:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"--window-csv not found: {path}")
        df = pd.read_csv(path)
        missing = {"ground_truth", "subject_id", "probability"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing {missing} -- not a p09k/p09f output CSV?")
        frames.append(df)
        print(f"Loaded {len(df)} window-level rows from {path}")
    return pd.concat(frames, ignore_index=True)


def plot_between_subject(subject_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """One panel per quantity: boxplot of each subject's own mean, patient vs. control, with
    individual subjects jittered on top so overlap/outliers/n are visible, not just the summary box."""
    specs = [s for s in FEATURE_SPECS if f"{s['col']}_mean" in subject_df.columns]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 4.2))
    if len(specs) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    for ax, spec in zip(axes, specs):
        col = f"{spec['col']}_mean"
        data_by_group = [subject_df.loc[subject_df["ground_truth"] == gt, col].dropna() for gt in (0, 1)]
        bp = ax.boxplot(data_by_group, tick_labels=[GROUP_STYLE[0]["label"], GROUP_STYLE[1]["label"]],
                         showfliers=False, widths=0.5, patch_artist=True)
        for patch, gt in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(GROUP_STYLE[gt]["color"])
            patch.set_alpha(0.25)
        for i, (gt, vals) in enumerate(zip((0, 1), data_by_group), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=GROUP_STYLE[gt]["color"],
                       alpha=0.6, s=18, zorder=3, edgecolors="none")
        ax.set_title(spec["label"], fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Between-subject: does the group difference show up broadly across the cohort?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_within_subject_shape(subject_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """One panel per quantity: each subject's own window-level percentiles, averaged within each
    group. Parallel, uniformly-separated lines across all percentiles = broad shift. Lines that
    hug together at low percentiles and only diverge near p90+ = a fat tail, not a broad shift."""
    pctl_cols_template = [f"p{p}" for p in PERCENTILES]
    specs = [s for s in FEATURE_SPECS if all(f"{s['col']}_{p}" in subject_df.columns for p in pctl_cols_template)]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 4.2))
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        cols = [f"{spec['col']}_{p}" for p in pctl_cols_template]
        for gt in (0, 1):
            sub = subject_df[subject_df["ground_truth"] == gt]
            if len(sub) == 0:
                continue
            means = sub[cols].mean().values
            style = GROUP_STYLE[gt]
            ax.plot(PERCENTILES, means, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], label=style["label"], markersize=5)
        ax.set_title(spec["label"], fontsize=10)
        ax.set_xlabel("Percentile", fontsize=9)
        ax.set_xticks(PERCENTILES)

    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(
        "Within-subject: is a typical subject's own recording broadly shifted, or tail-concentrated?",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def within_subject_median_spearman(windows: pd.DataFrame, x_col: str, y_col: str = "probability") -> float:
    """Median, across subjects, of that subject's own Spearman correlation between x_col and y_col --
    the same within-subject-first discipline used throughout this investigation (a pooled correlation
    across all windows/subjects would conflate within- and between-subject variance)."""
    per_subject_r = []
    for _subj_id, sub in windows.groupby("subject_id"):
        r = spearman_corr(sub[x_col].values, sub[y_col].values)
        if not np.isnan(r):
            per_subject_r.append(r)
    return float(np.median(per_subject_r)) if per_subject_r else float("nan")


def plot_window_level_relationship(
    windows: pd.DataFrame, output_path: Path, n_bins: int, scatter_sample: int, dpi: int,
) -> None:
    """One panel per band: does window_prob actually track real signal, including WITHIN each group
    separately? A pooled scatter across both groups can't distinguish "probability tracks band power"
    from "patients happen to differ on both quantities independently" -- fitting the trend separately
    within each group is what actually tests the former.

    Critically, the trend itself is computed on WITHIN-SUBJECT CENTERED values (each window's value
    minus that subject's own mean), not raw pooled values. Pooling raw values across subjects within
    a group reintroduces exactly the between-/within-subject conflation this investigation has
    repeatedly had to catch and correct elsewhere: with substantial subject-to-subject baseline
    spread (see between_subject.png), a raw pooled-and-binned trend mostly traces out WHICH SUBJECT a
    window came from, not how that subject's own probability moves with their own band power -- and
    can visually contradict the (correctly-computed) within-subject correlation annotated on the plot.
    Centering first makes the line consistent with what the annotation actually measures."""
    specs = [s for s in RELATIONSHIP_BANDS if s["col"] in windows.columns]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 4.6))
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        col = spec["col"]
        sub_all = windows[["subject_id", "ground_truth", col, "probability"]].dropna().copy()

        # Within-subject centering: subtract each subject's OWN mean from their own windows, for
        # both axes, before binning -- isolates within-subject covariation from between-subject
        # baseline differences, which is what the annotated statistic below actually measures.
        col_c = f"{col}_centered"
        sub_all[col_c] = sub_all[col] - sub_all.groupby("subject_id")[col].transform("mean")
        sub_all["probability_centered"] = (
            sub_all["probability"] - sub_all.groupby("subject_id")["probability"].transform("mean")
        )

        for gt in (0, 1):
            sub = sub_all[sub_all["ground_truth"] == gt]
            if len(sub) < n_bins:
                continue
            style = GROUP_STYLE[gt]

            if scatter_sample > 0:
                sample = sub.sample(n=min(scatter_sample, len(sub)), random_state=0)
                ax.scatter(sample[col_c], sample["probability_centered"], color=style["color"],
                           alpha=0.10, s=10, edgecolors="none", zorder=1)

            try:
                bins = pd.qcut(sub[col_c], q=n_bins, duplicates="drop")
            except ValueError:
                continue
            binned = sub.groupby(bins, observed=True).agg(
                x=(col_c, "mean"), y=("probability_centered", "mean")
            ).dropna()
            ax.plot(binned["x"], binned["y"], color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], label=style["label"], markersize=5, zorder=3, linewidth=2)

        r_patient = within_subject_median_spearman(sub_all[sub_all["ground_truth"] == 1], col)
        r_control = within_subject_median_spearman(sub_all[sub_all["ground_truth"] == 0], col)
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.text(
            0.02, 0.02,
            f"within-subject median r\npatient: {r_patient:+.2f}   control: {r_control:+.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="lightgray"),
        )
        ax.set_xlabel(f"{spec['label']} (within-subject centered)", fontsize=9)
        ax.set_ylabel("Model window probability (within-subject centered)", fontsize=9)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Window-level relationship: does probability track real signal WITHIN each group, not just between groups?",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_cli_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(args.window_csvs)
    feature_cols = [s["col"] for s in FEATURE_SPECS if s["col"] in windows.columns]
    missing = [s["col"] for s in FEATURE_SPECS if s["col"] not in windows.columns]
    if missing:
        print(f"[Warning] Columns not found in input CSV(s), skipping from figures: {missing}")

    subject_df = build_subject_level_summary(windows, feature_cols)
    print(f"Subject-level summary: {len(subject_df)} subjects "
          f"({int((subject_df['ground_truth'] == 1).sum())} patients, "
          f"{int((subject_df['ground_truth'] == 0).sum())} controls)")

    plot_between_subject(subject_df, output_dir / "between_subject.png", args.dpi)
    plot_within_subject_shape(subject_df, output_dir / "within_subject_shape.png", args.dpi)
    plot_window_level_relationship(windows, output_dir / "window_level_relationship.png",
                                    args.n_bins, args.scatter_sample, args.dpi)


if __name__ == "__main__":
    main()
