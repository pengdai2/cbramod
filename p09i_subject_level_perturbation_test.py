"""
p09i_subject_level_perturbation_test.py

The gap p09h left open: it perturbed one window at a time, holding every
other window of that subject fixed, and measured that window's OWN
probability response -- a real, valid window-level causal claim. But
subject-level classification comes from POOLING (p85_score by default)
across a subject's whole window set, and pooling is not a simple pass-
through: p09e's leave-one-out analysis already showed most windows have
near-zero contribution to a percentile statistic, only the ones near the
percentile rank matter. So a window-level causal effect doesn't
automatically compose into a subject-level effect on the pooled score
and hence classification.

This tests that composition directly: for each subject, perturb ALL of
their windows by the SAME scale factor simultaneously (not one at a
time), recompute every window's probability, then re-pool via the
model's actual pooling strategy -- tracing out a subject-level dose-
response curve (pooled_score vs. scale_factor) exactly analogous to
p09h's window-level one, just one level up.

Also reports, per subject:
  - The leave-one-out contribution profile of their baseline windows
    (reusing compute_leave_one_out_contributions from cbramod_common --
    no extra forward passes needed, it's pure math on the already-
    computed baseline probabilities), so you can see whether a subject
    with a MORE concentrated (few-window-dominated) pooling profile shows
    a bigger or smaller subject-level response than one with a diffuse
    profile.
  - A "propagation ratio": the pooled-score shift (between the lowest and
    highest tested scale factors) divided by the NAIVE mean window-level
    shift over the same range. A ratio near 1 means pooling passes
    through about as much of the window-level effect as a simple average
    would; a ratio well below 1 (or negative) is the concrete, per-
    subject signature of "most windows don't matter to this subject's
    pooled score," directly quantifying the gap this script exists to
    test.

Usage:
  python p09i_subject_level_perturbation_test.py \
      --checkpoint model.pt --manifest test_manifest.csv \
      --band sigma --scale-factors 0.5,0.75,1.0,1.25,1.5 \
      --subjects-json key_subjects.json --output-dir results/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cbramod_common import (
    BAND_DEFS,
    PANSubjectEEGDataset,
    add_log_filename_argument,
    build_e2e_classifier,
    compute_leave_one_out_contributions,
    compute_pooled_scores,
    extract_ckpt_metadata,
    get_operating_threshold,
    perturb_window_band_power,
    resolve_pooling_config,
    seed_everything,
    setup_inference_cli_parser,
    setup_perturbation_cli_parser,
    spearman_corr,
)
from cbramod_utils import setup_logger
from p09c_clinical_subject_diagnostics import load_subject_ids_from_json


def fit_local_slope(scale_factors: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """Linear regression of `values` (pooled score, here) on scale_factor -- returns (slope, R^2)."""
    if len(scale_factors) < 2 or np.std(scale_factors) == 0:
        return float("nan"), float("nan")
    A = np.vstack([scale_factors, np.ones_like(scale_factors)]).T
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((values - pred) ** 2)
    ss_tot = np.sum((values - values.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)



def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Subject-Level (Pooled) Band Power Perturbation Test")
    setup_perturbation_cli_parser(parser, output_csv_default="subject_level_perturbation.csv")
    perturb_group = parser.add_argument_group("Subject-Level Perturbation Test")
    perturb_group.add_argument(
        "--perturb-fraction", type=str, default="1.0",
        help="Comma-separated grid of fractions (each in (0, 1]) of a subject's sampled windows to "
             "actually perturb; the rest are left at their ORIGINAL baseline signal (never touched, "
             "regardless of scale_factor) for every point on the scale-factor sweep. Default '1.0' "
             "perturbs every window (the original design) -- one row per (subject, fraction) pair is "
             "written, and fractions below 1.0 are always a NESTED prefix of the same random per-subject "
             "permutation (fraction=0.25's perturbed set is a subset of fraction=0.5's, etc.), so the "
             "swept variable is cleanly 'how much of the recording changed' rather than also varying "
             "which specific random subset was drawn each time. Smaller fractions test whether the "
             "subject-level effect is fragile (only shows up when essentially everything changes) or "
             "robust (survives incomplete coverage) -- see the module docstring's note on why a per-"
             "window LOO-influence diagnostic was tried and dropped here (percentile pooling depends on "
             "the whole sorted distribution, not a fixed set of 'important' windows, so which windows "
             "specifically get perturbed can't be cleanly separated from how many do)."
    )
    add_log_filename_argument(parser, __file__)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./subject_level_perturbation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        raise ValueError("This script perturbs the raw waveform directly (--manifest); --cache-dir has "
                          "no raw signal left to filter.")

    low, high = BAND_DEFS[args.band]
    scale_factors = np.array([float(s) for s in args.scale_factors.split(",")])
    lo_scale_idx, hi_scale_idx = int(np.argmin(scale_factors)), int(np.argmax(scale_factors))
    perturb_fractions = sorted(float(f) for f in args.perturb_fraction.split(","))
    for f in perturb_fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"--perturb-fraction values must be in (0, 1], got {f}.")

    model, ckpt = build_e2e_classifier(args, device, logger)
    # Checked AFTER build_e2e_classifier(), not against the raw --num-classes flag: that flag is only
    # ever a cross-check/fallback (see resolve_checkpoint_architecture()), so a stale/default
    # --num-classes 2 alongside an actual 3-class checkpoint would otherwise silently pass this guard
    # and proceed to compute a meaningless softmax[:, 1] "positive probability" from a 3-way output,
    # instead of loudly rejecting it as this script requires.
    if ckpt.get("num_classes", args.num_classes) != 2:
        raise ValueError("This script assumes binary classification (softmax[:, 1] as 'the' probability).")
    ckpt_thresholds, _epoch, ckpt_pooling_params = extract_ckpt_metadata(ckpt)

    pooling_strategy, top_percentile, t_window = resolve_pooling_config(
        pooling_strategy=args.pooling_strategy, top_percentile=args.top_percentile,
        t_window=args.t_window, ckpt_pooling_params=ckpt_pooling_params
    )
    if pooling_strategy == "all":
        raise ValueError("--pooling-strategy all doesn't apply here -- pick one specific pooling formula.")
    threshold = get_operating_threshold(
        pooling_strategy=pooling_strategy, override_threshold=args.override_threshold, ckpt_thresholds=ckpt_thresholds
    )
    print(f"Pooling strategy: {pooling_strategy} (top_percentile={top_percentile}, t_window={t_window}), "
          f"operating threshold={threshold:.4f}")
    print(f"Perturbing band: {args.band} ({low}-{high} Hz). Scale factors: {scale_factors.tolist()}")

    subject_filter = [s.strip() for s in args.subject_id.split(",")] if args.subject_id else []
    if args.subjects_json:
        json_subject_ids = load_subject_ids_from_json(Path(args.subjects_json))
        print(f"Loaded {len(json_subject_ids)} subject_id(s) from --subjects-json: {args.subjects_json}")
        subject_filter = subject_filter + [sid for sid in json_subject_ids if sid not in subject_filter]
    subject_filter = subject_filter or None

    dataset = PANSubjectEEGDataset(
        manifest_csv=args.manifest, data_dir=args.data_dir, filter_stage=args.filter_stage,
        filter_subject=subject_filter, memory_map=True
    )
    print(f"Loaded raw EEG recording dataset for {len(dataset)} subjects.")

    rows: List[Dict] = []
    for idx in tqdm(range(len(dataset)), desc="Process Subjects"):
        x_tensor, y_tensor, subj_id, stages, indices = dataset[idx]
        num_windows = x_tensor.shape[0]
        if num_windows == 0:
            continue

        raw_np = x_tensor.numpy()  # [N, C, T]
        window_order = np.arange(num_windows)
        approximated = False
        if args.max_windows_per_subject and num_windows > args.max_windows_per_subject:
            window_order = rng.choice(num_windows, size=args.max_windows_per_subject, replace=False)
            approximated = True
        work_windows = raw_np[window_order]

        # Baseline (unperturbed) probabilities and pooled score -- computed ONCE per subject, reused
        # across every fraction in the sweep (baseline doesn't depend on perturb_fraction at all).
        with torch.no_grad():
            baseline_logits = model(torch.from_numpy(work_windows).to(device))
            baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].cpu().numpy()
        baseline_pooled = compute_pooled_scores(
            baseline_probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )

        # Leave-one-out contribution profile of the baseline windows -- pure math, no extra forward
        # passes, reused directly from p09e's implementation. Reported as general per-subject context
        # (how concentrated is this subject's OWN pooling sensitivity) -- NOT used to explain
        # perturb_fraction's effect, since percentile pooling depends on the whole sorted distribution,
        # not a fixed set of "important" windows identifiable from the baseline alone: perturbing even
        # nominally "uninfluential" windows can reshuffle the post-perturbation rank order enough to
        # move the percentile, so "did we hit the baseline-influential windows" doesn't cleanly explain
        # what happens once other windows change too.
        loo_contributions = compute_leave_one_out_contributions(
            baseline_probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )
        n_influential = int(np.sum(np.abs(loo_contributions) > 1e-6))

        # ONE random permutation per subject, reused (as nested prefixes) across every fraction in the
        # sweep -- so fraction=0.25's perturbed set is a subset of fraction=0.5's, etc. This isolates
        # "how much of the recording changed" as the swept variable; independently re-randomizing the
        # subset at each fraction would confound that with "which specific random subset got drawn."
        permutation = rng.permutation(len(work_windows))

        for frac in perturb_fractions:
            n_perturb = max(1, int(round(frac * len(work_windows))))
            perturb_mask = np.zeros(len(work_windows), dtype=bool)
            perturb_mask[permutation[:n_perturb]] = True

            # Perturb the selected subset simultaneously, once per scale factor, and re-pool each time.
            # Windows outside perturb_mask are passed through untouched -- exactly equivalent to
            # scale_factor=1.0 (an exact no-op per perturb_window_band_power's own docstring), so this
            # is just skipping needless recomputation.
            pooled_per_scale = np.zeros(len(scale_factors))
            mean_window_prob_per_scale = np.zeros(len(scale_factors))
            for i, s in enumerate(scale_factors):
                perturbed_batch = np.stack([
                    perturb_window_band_power(
                        w, args.sfreq, low, high, s, order=args.filter_order,
                        preserve_total_energy=args.preserve_total_energy
                    ) if perturb_mask[j] else w
                    for j, w in enumerate(work_windows)
                ])
                with torch.no_grad():
                    logits = model(torch.from_numpy(perturbed_batch).to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pooled_per_scale[i] = compute_pooled_scores(
                    probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
                )
                mean_window_prob_per_scale[i] = probs.mean()

            subject_slope, subject_r2 = fit_local_slope(scale_factors, pooled_per_scale)

            pooled_shift = pooled_per_scale[hi_scale_idx] - pooled_per_scale[lo_scale_idx]
            mean_window_shift = mean_window_prob_per_scale[hi_scale_idx] - mean_window_prob_per_scale[lo_scale_idx]
            propagation_ratio = float(pooled_shift / mean_window_shift) if abs(mean_window_shift) > 1e-9 else float("nan")

            rows.append({
                "subject_id": subj_id,
                "ground_truth": int(y_tensor.item()),
                "n_windows": len(work_windows),
                "perturb_fraction": frac,
                "n_windows_perturbed": int(n_perturb),
                "windows_approximated": approximated,
                "n_windows_with_nonzero_loo_contribution": n_influential,
                "baseline_pooled_score": float(baseline_pooled),
                "baseline_prediction": int(baseline_pooled >= threshold),
                "subject_level_slope": subject_slope,
                "subject_level_r2": subject_r2,
                "mean_window_level_shift": float(mean_window_shift),
                "pooled_score_shift": float(pooled_shift),
                "propagation_ratio": propagation_ratio,
            })

            print(
                f"  {subj_id} [frac={frac:.2f}]: baseline_pooled={baseline_pooled:.4f} | "
                f"subject_slope={subject_slope:+.4f} (R^2={subject_r2:.3f}) | "
                f"propagation_ratio={propagation_ratio:+.3f} | influential_windows={n_influential}/{len(work_windows)}"
            )

    if not rows:
        print("No subjects processed -- nothing to report.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} (subject, perturb_fraction) rows to: {csv_path}")

    report_fraction_sweep(df, perturb_fractions, band_label=args.band)


def report_fraction_sweep(df: pd.DataFrame, perturb_fractions: List[float], band_label: str = "the band") -> None:
    """
    Reports the subject-level slope/R^2/propagation_ratio summary, optionally broken down by
    perturb_fraction, plus (when more than one fraction is present) whether the subject-level result
    trends with how much of the recording actually changed. Factored out of main() so a separate
    aggregator script can call it on a combined DataFrame built from multiple already-completed runs
    (e.g. run at 1.0 and 0.5 separately, at different times) without needing to re-run anything.
    """
    print("\n" + "=" * 88)
    print(f"SUBJECT-LEVEL RESULT: does perturbing {band_label} across a whole recording move the pooled score?")
    print("=" * 88)
    if len(perturb_fractions) > 1:
        print(f"Broken down by perturb_fraction ({perturb_fractions}):\n")
        summary = df.groupby("perturb_fraction").agg(
            n_subjects=("subject_id", "count"),
            mean_slope=("subject_level_slope", "mean"),
            median_slope=("subject_level_slope", "median"),
            frac_neg=("subject_level_slope", lambda s: (s < 0).mean()),
            mean_r2=("subject_level_r2", "mean"),
            mean_propagation_ratio=("propagation_ratio", "mean"),
            median_propagation_ratio=("propagation_ratio", "median"),
        )
        print(summary.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.1f}"))
        print(
            "\nSlope is a RAW effect size, not normalized by how much of the recording was perturbed -- under "
            "a dilution/whole-distribution-shift model it is EXPECTED to scale down roughly proportionally "
            "with perturb_fraction (this is not evidence against the effect). What indicates the effect is "
            "robust (not an artifact of perturbing essentially the whole recording) is frac_neg and "
            "propagation_ratio (the fraction-normalized quantity) staying reasonably stable as perturb_fraction "
            "decreases, rather than collapsing toward 0.5 / 0."
        )
    else:
        print(f"  mean subject-level slope   = {df['subject_level_slope'].mean():+.4f}")
        print(f"  median subject-level slope = {df['subject_level_slope'].median():+.4f}")
        print(f"  frac(slope<0) = {(df['subject_level_slope'] < 0).mean():.2f}   "
              f"frac(slope>0) = {(df['subject_level_slope'] > 0).mean():.2f}")
        print(f"  mean subject-level fit R^2 = {df['subject_level_r2'].mean():.3f}")
        print(f"\n  mean propagation_ratio   = {df['propagation_ratio'].mean():+.4f}")
        print(f"  median propagation_ratio = {df['propagation_ratio'].median():+.4f}")
    if len(perturb_fractions) > 1:
        print(
            "\n  Note: mean_propagation_ratio can be noisy/unstable across fractions -- a subject with a "
            "small-but-nonzero mean_window_level_shift denominator produces an outsized ratio that skews the "
            "mean. median_propagation_ratio is the more robust summary to compare across perturb_fraction."
        )
    print(
        "\n  A ratio near 1.0 means the pooled score moves about as much as a naive window average would --"
        " pooling isn't losing much of the window-level effect. A ratio well below 1 (or negative) means"
        " most of the window-level signal is NOT reaching the subject-level decision, quantifying exactly"
        " the gap this script was built to test."
    )

    if len(perturb_fractions) > 1:
        print("\n" + "=" * 88)
        print("CORRELATION: does the subject-level result trend with how much of the recording changed?")
        print("=" * 88)
        print(
            "Pooled across every (subject, perturb_fraction) row -- NOT the per-window LOO-influence "
            "framing tried earlier (dropped: percentile pooling depends on the whole sorted distribution, "
            "not a fixed set of 'important' windows identifiable from the baseline alone, so 'did we hit "
            "the influential windows' doesn't cleanly separate from 'how much changed overall'). This is "
            "the clean version of that question: does MORE of the recording changing produce a stronger, "
            "more negative slope / higher propagation_ratio, in a graded, monotonic way?"
        )
        valid_slope = df["subject_level_slope"].notna()
        r_slope = spearman_corr(df.loc[valid_slope, "perturb_fraction"].values, df.loc[valid_slope, "subject_level_slope"].values)
        valid_ratio = df["propagation_ratio"].notna()
        r_ratio = spearman_corr(df.loc[valid_ratio, "perturb_fraction"].values, df.loc[valid_ratio, "propagation_ratio"].values)
        print(f"\n  Spearman(perturb_fraction, subject_level_slope)  = {r_slope:+.4f}  (n={int(valid_slope.sum())})")
        print(f"  Spearman(perturb_fraction, propagation_ratio)    = {r_ratio:+.4f}  (n={int(valid_ratio.sum())})")


if __name__ == "__main__":
    main()
