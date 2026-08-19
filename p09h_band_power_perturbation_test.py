"""
p09h_band_power_perturbation_test.py

The counterfactual/causal test this whole investigation has been building
toward: p09f found that relative band power (sigma in particular) predicts
a window's own probability, within-subject, replicated across two pools
and surviving a stage-confound check. But correlation can't distinguish
"the model relies on this feature" from "this feature happens to correlate
with something else the model actually uses." This script tests reliance
directly: synthetically scale a chosen frequency band's contribution to a
window's raw signal (uniformly across all channels, so the perturbation
targets the model's own channel-mean-pooled band power in a predictable,
linear way -- see `perturb_window_band_power`) and measure how the model's
own window-level probability moves in response.

For each sampled window, the target band is isolated via a zero-phase
Butterworth bandpass filter, then rescaled by a small grid of scale
factors (0.5x, 0.75x, 1x [original], 1.25x, 1.5x by default) while the
rest of the signal (everything outside the band) is left untouched. The
model is run on the original plus every perturbed copy, and a per-window
local slope (d(probability)/d(scale_factor)) is estimated via linear
regression across that grid -- along with the fit's R^2, since a low R^2
means the response isn't well-described by a straight line over this
range (the model's response could be more nonlinear/threshold-like there).

Critically, this also runs the baseline-dependence check discussed before
building this: does the measured slope depend on (a) the subject's own
baseline level of this band (the between-subject question that showed a
confusing/reversed pattern in p09g), and/or (b) how close the window's
baseline probability sits to the decision boundary (0.5) -- a generic
property of any sigmoid-like classifier, true for ANY perturbed feature,
not specific to this band at all. Both are reported via univariate
correlations AND a joint linear regression (slope ~ baseline_band_power +
distance_from_boundary) so one doesn't get credited for the other's
effect.

Usage:
  python p09h_band_power_perturbation_test.py \
      --checkpoint model.pt --manifest test_manifest.csv \
      --band sigma --scale-factors 0.5,0.75,1.0,1.25,1.5 \
      --max-windows-per-subject 40 --output-dir results/
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
    perturb_window_band_power,
    seed_everything,
    setup_inference_cli_parser,
    setup_perturbation_cli_parser,
    spearman_corr,
)
from cbramod_utils import setup_logger
from p09c_clinical_subject_diagnostics import load_subject_ids_from_json


def compute_relative_band_power(signal_1d: np.ndarray, sfreq: float, low: float, high: float) -> float:
    """Same relative-power definition as p09f_morphology_score_correlation.py, for a single band."""
    from scipy.signal import welch
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=min(len(signal_1d), int(sfreq * 4)))
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = psd[total_mask].sum()
    if total_power <= 0:
        return 0.0
    band_mask = (freqs >= low) & (freqs <= high)
    return float(psd[band_mask].sum() / total_power)


def fit_local_slope(scale_factors: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Linear regression of probability on scale_factor -- returns (slope, R^2)."""
    if len(scale_factors) < 2 or np.std(scale_factors) == 0:
        return float("nan"), float("nan")
    A = np.vstack([scale_factors, np.ones_like(scale_factors)]).T
    (slope, intercept), residuals, rank, sv = np.linalg.lstsq(A, probs, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((probs - pred) ** 2)
    ss_tot = np.sum((probs - probs.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def joint_regression(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> Dict[str, float]:
    """OLS of y ~ x1 + x2 (with intercept) -- disentangles two candidate explanations for `y`."""
    mask = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2))
    y, x1, x2 = y[mask], x1[mask], x2[mask]
    if len(y) < 4:
        return {"n": len(y), "beta_x1": float("nan"), "beta_x2": float("nan"), "r2": float("nan")}
    A = np.vstack([x1, x2, np.ones_like(y)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": len(y), "beta_x1": float(coeffs[0]), "beta_x2": float(coeffs[1]), "r2": float(r2)}


def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Band Power Perturbation (Counterfactual) Test")
    setup_perturbation_cli_parser(
        parser, output_csv_default="band_power_perturbation.csv", max_windows_per_subject_default=40,
    )
    add_log_filename_argument(parser, __file__)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./band_power_perturbation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        raise ValueError(
            "This script perturbs the raw waveform directly (--manifest); --cache-dir (pre-extracted "
            "embeddings) has no raw signal left to filter."
        )
    if args.num_classes != 2:
        raise ValueError("This script assumes binary classification (softmax[:, 1] as 'the' probability).")

    low, high = BAND_DEFS[args.band]
    scale_factors = np.array([float(s) for s in args.scale_factors.split(",")])
    if 1.0 not in scale_factors:
        print("  [Warning] --scale-factors doesn't include 1.0 (the unperturbed original) -- "
              "the reported 'baseline_probability' will still use the true original window, "
              "but the fitted slope won't be anchored at the actual unperturbed point.")

    model, _ckpt = build_e2e_classifier(args, device, logger)

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
    print(f"Perturbing band: {args.band} ({low}-{high} Hz). Scale factors: {scale_factors.tolist()}")

    all_rows: List[Dict] = []
    for idx in tqdm(range(len(dataset)), desc="Process Subjects"):
        x_tensor, y_tensor, subj_id, stages, indices = dataset[idx]
        num_windows = x_tensor.shape[0]
        if num_windows == 0:
            continue

        raw_np = x_tensor.numpy()  # [N, C, T]

        # Baseline (unperturbed) probability for every window, one batched forward pass.
        with torch.no_grad():
            baseline_logits = model(x_tensor.to(device))
            baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].cpu().numpy()

        # This subject's own baseline band-power level (mean across ALL their windows), for the
        # between-subject check -- computed once per subject, independent of which windows get perturbed.
        subject_band_powers = np.array([
            compute_relative_band_power(raw_np[i].mean(axis=0), args.sfreq, low, high)
            for i in range(num_windows)
        ])
        subject_mean_band_power = float(subject_band_powers.mean())

        window_order = np.arange(num_windows)
        if num_windows > args.max_windows_per_subject:
            window_order = rng.choice(num_windows, size=args.max_windows_per_subject, replace=False)

        for f_idx in window_order:
            window = raw_np[f_idx]  # [C, T]
            perturbed_batch = np.stack([
                perturb_window_band_power(
                    window, args.sfreq, low, high, s, order=args.filter_order,
                    preserve_total_energy=args.preserve_total_energy
                )
                for s in scale_factors
            ])
            with torch.no_grad():
                logits = model(torch.from_numpy(perturbed_batch).to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            slope, r2 = fit_local_slope(scale_factors, probs)

            all_rows.append({
                "subject_id": subj_id,
                "ground_truth": int(y_tensor.item()),
                "raw_epoch_index": int(indices[f_idx]),
                "stage": stages[f_idx] if f_idx < len(stages) else "UNKNOWN",
                "baseline_probability": float(baseline_probs[f_idx]),
                "distance_from_boundary": float(abs(baseline_probs[f_idx] - 0.5)),
                "window_band_power": float(subject_band_powers[f_idx]),
                "subject_mean_band_power": subject_mean_band_power,
                "slope": slope,
                "slope_r2": r2,
            })

    if not all_rows:
        print("No windows collected -- nothing to analyze.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} perturbed-window results across {df['subject_id'].nunique()} subjects to: {csv_path}")

    valid = df.dropna(subset=["slope"])
    print("\n" + "=" * 88)
    print(f"PRIMARY RESULT: sign/magnitude of the local slope (d(probability)/d(scale_factor) for {args.band})")
    print("=" * 88)
    print(f"  n windows with a valid slope: {len(valid)} / {len(df)}")
    print(f"  mean slope   = {valid['slope'].mean():+.4f}")
    print(f"  median slope = {valid['slope'].median():+.4f}")
    print(f"  frac(slope<0) = {(valid['slope'] < 0).mean():.2f}   frac(slope>0) = {(valid['slope'] > 0).mean():.2f}")
    print(f"  mean fit R^2  = {valid['slope_r2'].mean():.3f}  (low R^2 -> response isn't well-described by a "
          f"straight line over this scale-factor range for that window)")

    print("\n" + "=" * 88)
    print("BASELINE-DEPENDENCE CHECK: does the slope depend on subject baseline power, or boundary proximity?")
    print("=" * 88)
    r_power = spearman_corr(valid["slope"].values, valid["subject_mean_band_power"].values)
    r_boundary = spearman_corr(valid["slope"].values, valid["distance_from_boundary"].values)
    print(f"  Spearman(slope, subject_mean_band_power) = {r_power:+.4f}  (univariate -- between-subject baseline)")
    print(f"  Spearman(slope, distance_from_boundary)  = {r_boundary:+.4f}  (univariate -- generic classifier sensitivity)")

    joint = joint_regression(
        valid["slope"].values, valid["subject_mean_band_power"].values, valid["distance_from_boundary"].values
    )
    print(
        f"  Joint OLS  slope ~ subject_mean_band_power + distance_from_boundary  (n={joint['n']}, R^2={joint['r2']:.3f}):\n"
        f"    beta(subject_mean_band_power)   = {joint['beta_x1']:+.4f}\n"
        f"    beta(distance_from_boundary)    = {joint['beta_x2']:+.4f}\n"
        f"  If |beta(distance_from_boundary)| dominates once both are in the model together, the "
        f"univariate baseline-power correlation above was likely riding on boundary proximity, not "
        f"the band's baseline level specifically -- and vice versa."
    )


if __name__ == "__main__":
    main()
