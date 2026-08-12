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
    CBraModE2EClassifier,
    PANSubjectEEGDataset,
    compute_leave_one_out_contributions,
    compute_pooled_scores,
    get_operating_threshold,
    load_model_checkpoint,
    perturb_window_band_power,
    resolve_pooling_config,
    setup_inference_cli_parser
)
from cbramod_common import seed_everything
from p09c_clinical_subject_diagnostics import load_subject_ids_from_json

BAND_DEFS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}


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


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Subject-Level (Pooled) Band Power Perturbation Test")
    group = parser.add_argument_group("Subject-Level Perturbation Test")
    group.add_argument(
        "--band", type=str, default="sigma", choices=list(BAND_DEFS.keys()),
        help="Which frequency band to perturb (default: sigma)."
    )
    group.add_argument(
        "--scale-factors", type=str, default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated grid of scale factors applied to the band's amplitude, in EVERY window "
             "of a subject simultaneously (1.0 = unperturbed original)."
    )
    group.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order for band isolation.")
    group.add_argument(
        "--no-preserve-total-energy", dest="preserve_total_energy", action="store_false",
        help="Disable renormalizing each perturbed channel back to its original std after the band "
             "rescale (see perturb_window_band_power()'s docstring). Default: preserve_total_energy=True."
    )
    group.add_argument(
        "--max-windows-per-subject", type=int, default=None,
        help="Optional cap on windows perturbed per subject, for speed on very long recordings. The "
             "GPU forward pass batches all sampled windows together (one pass per scale factor "
             "regardless of window count), but the CPU-side Butterworth filtering does NOT -- it's "
             "vectorized across channels but still runs once per window per scale factor, so it scales "
             "linearly with window count and is the dominant cost for a full night (num_windows x "
             "len(scale_factors) filter calls per subject). Default: use every valid window (no cap); "
             "set this (e.g. 100-200) if runtime matters more than perturbing literally every window. "
             "NOTE: capping means only a SUBSET of windows get perturbed while the rest keep their "
             "original probability -- an approximation of 'the whole recording changed', not the real thing."
    )
    group.add_argument(
        "--perturb-fraction", type=float, default=1.0,
        help="Fraction (0-1] of a subject's sampled windows to actually perturb; the rest are left at "
             "their ORIGINAL baseline signal (never touched, regardless of scale_factor) for every "
             "point on the scale-factor sweep. Default 1.0 perturbs every window (the original design). "
             "A smaller fraction (e.g. 0.5) is a sharper test of the specific concern p09e's leave-one-"
             "out analysis raised: percentile pooling only cares about whichever window sits at the "
             "rank, and most windows have near-zero contribution to it. Perturbing ALL windows sidesteps "
             "that -- even a window that individually doesn't matter to the percentile still shifts, "
             "which can move the whole distribution enough that something near the rank moves too, "
             "without ever testing whether pooling ignores a change concentrated in the 'wrong' "
             "(non-rank-adjacent) windows. Perturbing a random, fixed-across-the-sweep subset instead "
             "directly tests that: if the percentile-determining window happens to land in the "
             "UNPERTURBED remainder for a given subject, the pooled score should barely move despite a "
             "real average shift across the perturbed subset -- exactly the failure mode p09e predicts, "
             "and something --perturb-fraction=1.0 can't distinguish from faithful propagation."
    )
    group.add_argument(
        "--subjects-json", type=str, default=None,
        help="Path to a p09d_subject_confidence_report.py --output-json report. Its subject_ids are "
             "unioned with --subject-id (if also given) to select which subjects to analyze."
    )
    group.add_argument(
        "--output-csv", type=str, default="subject_level_perturbation.csv",
        help="Filename (relative to --output-dir) for the per-subject results."
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./subject_level_perturbation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.features_pt:
        raise ValueError("This script perturbs the raw waveform directly (--manifest); --features-pt has "
                          "no raw signal left to filter.")
    if args.num_classes != 2:
        raise ValueError("This script assumes binary classification (softmax[:, 1] as 'the' probability).")

    low, high = BAND_DEFS[args.band]
    scale_factors = np.array([float(s) for s in args.scale_factors.split(",")])
    lo_scale_idx, hi_scale_idx = int(np.argmin(scale_factors)), int(np.argmax(scale_factors))

    print("Instantiating full CBraModE2EClassifier for raw waveform inference.")
    model = CBraModE2EClassifier(
        num_channels=args.num_channels, sfreq=args.sfreq, num_patches=args.num_patches,
        emb_dim=args.cbra_dim, hidden_dim=args.head_dim, num_classes=args.num_classes,
        head_type=args.head_type
    )
    model, ckpt_thresholds, _, ckpt_pooling_params = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

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

        # Which of work_windows actually get perturbed -- a FIXED subset held constant across the
        # whole scale-factor sweep (not re-drawn per scale factor, or the fitted slope would be
        # confounded by ALSO changing which windows are perturbed at each point). Windows outside this
        # subset are left at their true, untouched baseline signal, but still participate in pooling --
        # unlike --max-windows-per-subject, which drops excluded windows from pooling entirely, this
        # keeps the full real window distribution intact and tests whether the percentile statistic
        # responds to a change concentrated in only PART of it. See --perturb-fraction's help text.
        n_perturb = max(1, int(round(args.perturb_fraction * len(work_windows))))
        perturb_mask = np.zeros(len(work_windows), dtype=bool)
        perturb_mask[rng.choice(len(work_windows), size=n_perturb, replace=False)] = True

        # Baseline (unperturbed) probabilities and pooled score.
        with torch.no_grad():
            baseline_logits = model(torch.from_numpy(work_windows).to(device))
            baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].cpu().numpy()
        baseline_pooled = compute_pooled_scores(
            baseline_probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )

        # Leave-one-out contribution profile of the baseline windows -- pure math, no extra forward
        # passes, reused directly from p09e's implementation.
        loo_contributions = compute_leave_one_out_contributions(
            baseline_probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )
        influential_mask = np.abs(loo_contributions) > 1e-6
        n_influential = int(np.sum(influential_mask))
        # Diagnostic for --perturb-fraction < 1.0: did the randomly-chosen perturbed subset happen to
        # include the windows that actually matter to the pooling statistic, or miss them? Directly
        # explains why a given subject's propagation_ratio comes out high or low under partial
        # perturbation, rather than leaving it as an unexplained per-subject coincidence.
        n_influential_perturbed = int(np.sum(influential_mask & perturb_mask))

        # Perturb the selected subset (ALL sampled windows, if --perturb-fraction=1.0, the default)
        # simultaneously, once per scale factor, and re-pool each time. Windows outside perturb_mask
        # are passed through untouched -- exactly equivalent to scale_factor=1.0 (an exact no-op per
        # perturb_window_band_power's own docstring), so this is just skipping needless recomputation.
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
            "n_windows_perturbed": int(n_perturb),
            "windows_approximated": approximated,
            "n_windows_with_nonzero_loo_contribution": n_influential,
            "n_influential_windows_perturbed": n_influential_perturbed,
            "baseline_pooled_score": float(baseline_pooled),
            "baseline_prediction": int(baseline_pooled >= threshold),
            "subject_level_slope": subject_slope,
            "subject_level_r2": subject_r2,
            "mean_window_level_shift": float(mean_window_shift),
            "pooled_score_shift": float(pooled_shift),
            "propagation_ratio": propagation_ratio,
        })

        print(
            f"  {subj_id}: baseline_pooled={baseline_pooled:.4f} | subject_slope={subject_slope:+.4f} "
            f"(R^2={subject_r2:.3f}) | propagation_ratio={propagation_ratio:+.3f} | "
            f"influential_windows={n_influential}/{len(work_windows)}"
        )

    if not rows:
        print("No subjects processed -- nothing to report.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} subjects' results to: {csv_path}")

    print("\n" + "=" * 88)
    print(f"SUBJECT-LEVEL RESULT: does perturbing {args.band} across a whole recording move the pooled score?")
    print("=" * 88)
    print(f"  mean subject-level slope   = {df['subject_level_slope'].mean():+.4f}")
    print(f"  median subject-level slope = {df['subject_level_slope'].median():+.4f}")
    print(f"  frac(slope<0) = {(df['subject_level_slope'] < 0).mean():.2f}   "
          f"frac(slope>0) = {(df['subject_level_slope'] > 0).mean():.2f}")
    print(f"  mean subject-level fit R^2 = {df['subject_level_r2'].mean():.3f}")
    print(f"\n  mean propagation_ratio   = {df['propagation_ratio'].mean():+.4f}")
    print(f"  median propagation_ratio = {df['propagation_ratio'].median():+.4f}")
    print(
        "  A ratio near 1.0 means the pooled score moves about as much as a naive window average would --"
        " pooling isn't losing much of the window-level effect. A ratio well below 1 (or negative) means"
        " most of the window-level signal is NOT reaching the subject-level decision, quantifying exactly"
        " the gap this script was built to test."
    )

    if args.perturb_fraction < 1.0:
        print("\n" + "=" * 88)
        print(f"PARTIAL-PERTURBATION DIAGNOSTIC (--perturb-fraction={args.perturb_fraction})")
        print("=" * 88)
        print(
            "Does propagation_ratio depend on whether the randomly-perturbed subset happened to include "
            "the windows that actually matter to pooling (nonzero LOO contribution), or missed them? "
            "This is the direct test of p09e's concern that --perturb-fraction=1.0 can't distinguish "
            "from faithful propagation."
        )
        has_influential = df["n_windows_with_nonzero_loo_contribution"] > 0
        frac_influential_perturbed = np.where(
            has_influential,
            df["n_influential_windows_perturbed"] / df["n_windows_with_nonzero_loo_contribution"].replace(0, np.nan),
            np.nan
        )
        valid = has_influential & df["propagation_ratio"].notna()
        r = spearman_corr(frac_influential_perturbed[valid], df.loc[valid, "propagation_ratio"].values)
        print(
            f"\n  Spearman(frac_influential_windows_perturbed, propagation_ratio) = {r:+.4f} "
            f"(n_subjects={int(valid.sum())})"
        )
        print(
            "  A positive correlation here would confirm: subjects where perturbation missed the "
            "influential windows show a lower propagation_ratio, and subjects where it hit them show a "
            "higher one -- i.e. the partial-perturbation result is explained by WHICH windows changed, "
            "not just how many."
        )


if __name__ == "__main__":
    main()
