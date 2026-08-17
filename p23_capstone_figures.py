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
                                 Answers "is there a broad, cohort-wide group difference?" Each
                                 subject sits at the SAME horizontal slot in every panel (ordered by
                                 their own model probability), so one subject's profile can be
                                 tracked across quantities by eye. If a --stratification-csv (p21's
                                 output) is supplied, subjects model[0] misclassifies are marked with
                                 a diamond AND a small number (stable across panels and shared with
                                 Figure 3) -- numbers rather than connecting lines, which get visually
                                 overwhelming fast.
  2. within_subject_shape.png -- for each percentile (p10-p99), the MEAN ACROSS SUBJECTS of that
                                 subject's own percentile value -- i.e. average-of-percentiles, not a
                                 single representative "typical" subject and not a median-of-subjects.
                                 Answers "is a group's typical shift broad, or tail-concentrated?"
  3. window_level_relationship.png -- one point per subject at their own TRUE (mean band power,
                                 mean probability), with a short tangent line through it showing
                                 that subject's OWN within-subject slope (simple OLS fit of
                                 probability on band power, using only that subject's own windows;
                                 tangent length is a FIXED value shared by every subject in the
                                 panel -- the median subject's own data spread -- not that subject's
                                 actual range, chosen so a few wide-spread subjects don't visually
                                 dominate). Point position shows the between-subject offset (should
                                 match between_subject.png); tangent direction shows the within-
                                 subject relationship -- deliberately kept separate rather than
                                 collapsed into one pooled or fully-centered trend line, which can
                                 only show one of these two facts at a time and risks looking like
                                 it denies the other. Misclassified subjects (same numbering as
                                 Figure 1) are marked here too.

Usage:
    python p23_capstone_figures.py \
        --window-csv val_ckpt/absolute_band_power_analysis.csv \
        --window-csv test_ckpt/absolute_band_power_analysis.csv \
        --stratification-csv model0_confidence_stratification.csv \
        --output-dir figures/

--stratification-csv is optional -- it's p21_model0_confidence_stratification.py's own already-saved
output, read here only to flag which subjects it marked is_correct == False. No new model inference
or threshold computation happens in this script either way (the threshold was already applied once,
by p21, using model[0]'s own calibrated value from its checkpoint).
"""

import argparse
from pathlib import Path
from typing import List, Optional

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
    parser.add_argument(
        "--stratification-csv", action="append", default=None, dest="stratification_csvs",
        help="Optional: path to p21_model0_confidence_stratification.py's output CSV (repeatable). "
             "If given, subjects with is_correct == False are flagged as misclassified in "
             "between_subject.png. Never used to compute anything, just to read an already-decided "
             "flag off an already-calibrated threshold."
    )
    parser.add_argument("--output-dir", type=str, default="figures")
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


def load_stratification(stratification_csvs: Optional[List[str]]) -> Optional[pd.DataFrame]:
    """Reads p21's already-saved per-subject output, if given -- just the is_correct flag (and
    category, if present), never recomputed here. Returns None if no CSV was passed, so callers can
    skip misclassification flagging entirely rather than fabricating a flag from scratch."""
    if not stratification_csvs:
        return None
    frames = []
    for csv_path in stratification_csvs:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"--stratification-csv not found: {path}")
        df = pd.read_csv(path)
        missing = {"subject_id", "is_correct"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing {missing} -- not a p21 output CSV?")
        cols = ["subject_id", "is_correct"] + (["category"] if "category" in df.columns else [])
        frames.append(df[cols])
        print(f"Loaded {len(df)} subject rows from {path}")
    combined = pd.concat(frames, ignore_index=True)
    dup = combined["subject_id"].duplicated()
    if dup.any():
        print(f"[Warning] {int(dup.sum())} duplicate subject_id rows across --stratification-csv "
              f"inputs -- keeping the first occurrence of each.")
        combined = combined.drop_duplicates(subset="subject_id", keep="first")
    return combined


def assign_misclassified_numbers(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'misclassified_number' column (1, 2, 3, ... for subjects with is_correct == False, NaN
    for everyone else), in a stable subject_id-sorted order -- shared between Figure 1 and Figure 3
    so the same subject carries the same number in both, without needing to draw a connecting line
    between figures (or, within Figure 1, across panels -- see plot_between_subject)."""
    if "is_correct" not in subject_df.columns:
        subject_df = subject_df.copy()
        subject_df["misclassified_number"] = np.nan
        return subject_df
    subject_df = subject_df.copy()
    mis_ids = sorted(subject_df.loc[subject_df["is_correct"] == False, "subject_id"])
    numbering = {sid: i + 1 for i, sid in enumerate(mis_ids)}
    subject_df["misclassified_number"] = subject_df["subject_id"].map(numbering)
    if numbering:
        print("Misclassified subject numbering (shared across figures):")
        for sid, num in numbering.items():
            row = subject_df.loc[subject_df["subject_id"] == sid].iloc[0]
            cat = f", {row['category']}" if "category" in subject_df.columns and pd.notna(row.get("category")) else ""
            print(f"  #{num}: {sid} (ground_truth={int(row['ground_truth'])}{cat})")
    return subject_df


def compute_subject_jitter_positions(subject_df: pd.DataFrame, width: float = 0.12) -> pd.Series:
    """One FIXED x-offset per subject, ordered by that subject's own model window probability,
    reused identically across every panel -- so the same subject sits at the same relative
    horizontal slot in every subplot. This is what lets a viewer visually track one subject's
    profile across quantities (without needing a connecting line for every subject, which would be
    overwhelming with this many points) -- the misclassified subjects below get an explicit
    connecting line on top of this, since they're the few actually worth tracing individually."""
    order_col = "probability_mean" if "probability_mean" in subject_df.columns else "subject_id"
    offsets = pd.Series(0.0, index=subject_df.index)
    for gt in (0, 1):
        idx = subject_df.index[subject_df["ground_truth"] == gt]
        if len(idx) == 0:
            continue
        ordered = subject_df.loc[idx, order_col].sort_values().index
        positions = np.linspace(-width, width, len(ordered)) if len(ordered) > 1 else np.array([0.0])
        offsets.loc[ordered] = positions
    return offsets


def plot_between_subject(subject_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """One panel per quantity: boxplot of each subject's own mean, patient vs. control, with
    individual subjects jittered on top so overlap/outliers/n are visible, not just the summary box.

    Two additions on top of the basic box+jitter:
      - Each subject gets a FIXED x-offset (compute_subject_jitter_positions), identical in every
        panel, so the same subject can be visually tracked across quantities by horizontal position
        alone.
      - Subjects model[0] misclassifies (subject_df["misclassified_number"] not NaN, from
        assign_misclassified_numbers) are drawn as a distinct diamond marker with a small number next
        to it in every panel -- a small numeric label rather than a connecting line across panels,
        which gets visually overwhelming fast with more than a couple of flagged subjects."""
    specs = [s for s in FEATURE_SPECS if f"{s['col']}_mean" in subject_df.columns]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 4.4))
    if len(specs) == 1:
        axes = [axes]

    jitter = compute_subject_jitter_positions(subject_df)
    has_strat = "is_correct" in subject_df.columns
    misclassified_mask = subject_df["misclassified_number"].notna() if "misclassified_number" in subject_df.columns else pd.Series(False, index=subject_df.index)
    n_misclassified = int(misclassified_mask.sum())

    for ax, spec in zip(axes, specs):
        col = f"{spec['col']}_mean"

        for gt in (0, 1):
            group_idx = subject_df.index[(subject_df["ground_truth"] == gt) & subject_df[col].notna()]
            x_pos = pd.Series((1 if gt == 0 else 2), index=group_idx) + jitter.loc[group_idx]
            vals = subject_df.loc[group_idx, col]

            normal_idx = group_idx[~misclassified_mask.loc[group_idx]]
            mis_idx = group_idx[misclassified_mask.loc[group_idx]]

            ax.scatter(x_pos.loc[normal_idx], vals.loc[normal_idx], color=GROUP_STYLE[gt]["color"],
                       alpha=0.6, s=18, zorder=3, edgecolors="none")
            if len(mis_idx) > 0:
                ax.scatter(x_pos.loc[mis_idx], vals.loc[mis_idx], color=GROUP_STYLE[gt]["color"],
                           alpha=0.95, s=70, zorder=5, marker="D", edgecolors="black", linewidths=1.2)
                for idx in mis_idx:
                    num = int(subject_df.loc[idx, "misclassified_number"])
                    ax.annotate(str(num), (x_pos.loc[idx], vals.loc[idx]), textcoords="offset points",
                                xytext=(6, 5), fontsize=7.5, fontweight="bold", zorder=6)

        data_by_group = [subject_df.loc[subject_df["ground_truth"] == gt, col].dropna() for gt in (0, 1)]
        bp = ax.boxplot(data_by_group, tick_labels=[GROUP_STYLE[0]["label"], GROUP_STYLE[1]["label"]],
                         showfliers=False, widths=0.5, patch_artist=True, zorder=1)
        for patch, gt in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(GROUP_STYLE[gt]["color"])
            patch.set_alpha(0.25)

        ax.set_title(spec["label"], fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

    if n_misclassified:
        misclass_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="gray",
                                      markeredgecolor="black", markersize=8, linestyle="none",
                                      label=f"Misclassified by model[0], numbered (n={n_misclassified})")
        axes[0].legend(handles=[misclass_handle], fontsize=8, loc="upper left")

    subtitle = "Between-subject: does the group difference show up broadly across the cohort?"
    if has_strat:
        subtitle += " (same subject = same slot in every panel; numbered diamonds = misclassified)"
    fig.suptitle(subtitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}" + (f" ({n_misclassified} misclassified subjects flagged)" if has_strat else ""))


def plot_within_subject_shape(subject_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """One panel per quantity. Each line point is the MEAN, ACROSS SUBJECTS IN THAT GROUP, of that
    subject's own percentile value (e.g. the p75 point = the average of every subject's own p75,
    each computed from that subject's own windows first) -- explicitly NOT a median-of-subjects, and
    NOT one single representative "typical" subject; it's an average-of-percentiles, computed
    subject-first so it never pools raw windows across subjects (see Chapter 10 of the writeup).
    Parallel, uniformly-separated lines across all percentiles = broad shift. Lines that hug
    together at low percentiles and only diverge near p90+ = a fat tail, not a broad shift.
    Being a mean, this is somewhat sensitive to outlier subjects, same as any mean would be."""
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
        "Within-subject: MEAN ACROSS SUBJECTS of each subject's own percentile -- broad shift or tail-concentrated?",
        fontsize=10.5,
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


def per_subject_slope_stats(windows: pd.DataFrame, x_col: str, y_col: str = "probability") -> pd.DataFrame:
    """Per subject: that subject's own TRUE (mean_x, mean_y) location, plus the LOCAL slope of y on
    x fit using ONLY that subject's own windows (simple OLS), plus that subject's own SD of x (used
    to size the tangent segment drawn through their point). These are the two pieces of information
    Figure 3 needs to show together without conflating them: WHERE a subject sits (the between-
    subject offset, which should match between_subject.png) and WHICH WAY their own windows tilt
    (the within-subject slope) -- a single pooled or fully-centered trend line can only show one of
    these at a time."""
    rows = []
    for subject_id, sub in windows.groupby("subject_id"):
        x = sub[x_col].to_numpy(dtype=np.float64)
        y = sub[y_col].to_numpy(dtype=np.float64)
        if len(x) < 5 or np.std(x) == 0:
            continue
        slope, _intercept = np.polyfit(x, y, 1)
        rows.append({
            "subject_id": subject_id,
            "ground_truth": int(sub["ground_truth"].iloc[0]),
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
            "sd_x": float(np.std(x)),
            "slope": float(slope),
        })
    return pd.DataFrame(rows)


def plot_window_level_relationship(windows: pd.DataFrame, subject_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """One panel per band. Each subject is ONE point at their own true (mean band power, mean
    probability) -- recovering the between-subject offset, which should visually match
    between_subject.png -- with a short tangent line through it showing that subject's OWN
    within-subject slope: a plain OLS fit of probability on band power using ONLY that subject's
    own windows (per_subject_slope_stats). Deliberately NOT a single pooled or within-subject-
    centered trend line: centering alone shows the slope but erases the group offset (looks like it
    denies a difference between_subject.png already shows); pooling raw values without centering
    reintroduces the between-/within-subject conflation this investigation has repeatedly had to
    catch elsewhere. Plotting both facts side by side, without collapsing them into one number,
    avoids having to choose which one to hide.

    The tangent's LENGTH is not that subject's own data range -- it's a single FIXED half-length
    (the median subject's own sd_x in this panel) applied to every subject, so a few subjects with
    unusually wide within-subject spread don't get long, visually dominant lines that drown out the
    pattern; only the tangent's DIRECTION (slope) is subject-specific. The bold tangent per group is
    the same idea at the group centroid, using that group's median per-subject slope -- the single
    clearest signal in the panel, layered on top of (not replacing) the individually noisy ticks.

    Misclassified subjects (subject_df["misclassified_number"], shared numbering with Figure 1) are
    drawn with the same diamond-plus-number marking used there."""
    numbering = subject_df.set_index("subject_id")["misclassified_number"] if "misclassified_number" in subject_df.columns else pd.Series(dtype=float)

    specs = [s for s in RELATIONSHIP_BANDS if s["col"] in windows.columns]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.6 * len(specs), 4.8))
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        col = spec["col"]
        clean = windows[["subject_id", "ground_truth", col, "probability"]].dropna()
        stats_df = per_subject_slope_stats(clean, col)
        stats_df["misclassified_number"] = stats_df["subject_id"].map(numbering)

        # A FIXED tangent half-length (not each subject's own sd_x) so a handful of subjects with
        # unusually wide within-subject spread don't get visually dominant, sprawling lines that
        # drown out the pattern -- individual per-subject slopes are already high-variance
        # estimates from ~100-200 noisy windows each; letting a few of them stretch across most of
        # the x-axis compounds noise with visual clutter. Sized off the median subject's spread.
        tick_half_length = float(stats_df["sd_x"].median()) if len(stats_df) else 1.0

        for gt in (0, 1):
            sub = stats_df[stats_df["ground_truth"] == gt]
            if len(sub) == 0:
                continue
            style = GROUP_STYLE[gt]
            normal = sub[sub["misclassified_number"].isna()]
            mis = sub[sub["misclassified_number"].notna()]

            ax.scatter(normal["mean_x"], normal["mean_y"], color=style["color"], alpha=0.85, s=30,
                       zorder=4, label=style["label"], edgecolors="white", linewidths=0.6)
            if len(mis) > 0:
                ax.scatter(mis["mean_x"], mis["mean_y"], color=style["color"], alpha=0.95, s=75,
                           zorder=6, marker="D", edgecolors="black", linewidths=1.2)
                for _, row in mis.iterrows():
                    ax.annotate(str(int(row["misclassified_number"])), (row["mean_x"], row["mean_y"]),
                                textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                                fontweight="bold", zorder=7)
            for _, row in sub.iterrows():
                half = tick_half_length
                x0, x1 = row["mean_x"] - half, row["mean_x"] + half
                y0, y1 = row["mean_y"] - row["slope"] * half, row["mean_y"] + row["slope"] * half
                ax.plot([x0, x1], [y0, y1], color=style["color"], alpha=0.35, linewidth=1.0, zorder=2)

            # One bold "group average" tangent at the group's own centroid, using the group's
            # median per-subject slope -- the single clearest signal in the panel, drawn on top of
            # the individually-noisy per-subject ticks rather than in place of them.
            centroid_x, centroid_y = sub["mean_x"].mean(), sub["mean_y"].mean()
            group_slope = float(sub["slope"].median())
            half = tick_half_length * 2.5
            gx0, gx1 = centroid_x - half, centroid_x + half
            gy0, gy1 = centroid_y - group_slope * half, centroid_y + group_slope * half
            ax.plot([gx0, gx1], [gy0, gy1], color=style["color"], alpha=1.0, linewidth=3.5,
                    zorder=5, solid_capstyle="round")

        r_patient = within_subject_median_spearman(clean[clean["ground_truth"] == 1], col)
        r_control = within_subject_median_spearman(clean[clean["ground_truth"] == 0], col)
        ax.text(
            0.02, 0.02,
            f"within-subject median r\npatient: {r_patient:+.2f}   control: {r_control:+.2f}\n"
            f"(dot = subject mean; faint tick = that subject's own slope,\n"
            f"drawn at a FIXED shared length, not their own data range;\n"
            f"bold tick = group median slope at group centroid;\n"
            f"numbered diamond = misclassified by model[0])",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="lightgray"),
        )
        ax.set_xlabel(spec["label"], fontsize=9)
        ax.set_ylabel("Model window probability", fontsize=9)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Window-level relationship: between-subject offset (point position) vs. within-subject slope (tick direction)",
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

    strat_df = load_stratification(args.stratification_csvs)
    if strat_df is not None:
        subject_df = subject_df.merge(strat_df, on="subject_id", how="left")
        n_matched = subject_df["is_correct"].notna().sum()
        n_unmatched = len(subject_df) - n_matched
        print(f"Merged stratification data: {n_matched} subjects matched"
              + (f", {n_unmatched} had no match in --stratification-csv (not flagged)" if n_unmatched else ""))
    else:
        print("No --stratification-csv given -- misclassified subjects will not be flagged "
              "(see p21_model0_confidence_stratification.py to generate one).")
    subject_df = assign_misclassified_numbers(subject_df)

    plot_between_subject(subject_df, output_dir / "between_subject.png", args.dpi)
    plot_within_subject_shape(subject_df, output_dir / "within_subject_shape.png", args.dpi)
    plot_window_level_relationship(windows, subject_df, output_dir / "window_level_relationship.png", args.dpi)


if __name__ == "__main__":
    main()
