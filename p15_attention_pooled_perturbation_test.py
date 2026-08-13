"""
p15_attention_pooled_perturbation_test.py

Direct follow-up to a sharp question raised while interpreting p13/p14's attention-MIL gate: the
subject-level score under attention pooling is `sum(attn_weight_i * window_prob_i)`, and
`window_prob_i` comes from the FROZEN window-level probe -- the same probe whose window-level
probability was already shown (p09h/p09i) to respond causally to sigma-band perturbation. p14's
attn_weight-vs-band-power correlations only test whether the ATTENTION MECHANISM ITSELF
independently tracks a band; they say nothing about whether that band's effect still reaches the
final attention-POOLED score, since that effect could travel entirely through window_prob without
attn_weight needing to "rediscover" it. This script tests that directly: does the validated sigma
causal effect still propagate to the subject-level decision once p85 percentile pooling is replaced
by the trained attention head?

Mirrors p09i_subject_level_perturbation_test.py's perturbation mechanics exactly (same raw-waveform
perturbation, same scale-factor sweep, same perturb_fraction design) but computes BOTH the p85-pooled
score and the attention-pooled score from the IDENTICAL perturbed window_probs at every step, so any
difference between them is attributable only to the pooling rule, not to any difference in the
underlying signal or window-level predictions.

Mechanically, this requires decomposing the fused CBraModE2EClassifier forward pass
(backbone -> mean-pool -> head) into its two stages, since the attention head needs the intermediate
per-window embedding (not just the final window_prob) as its own input -- p09i's `model(x)` call only
ever returns the final logits.

Usage:
    python p15_attention_pooled_perturbation_test.py \
        --checkpoint checkpoints-probe-linear/cbramod_ckpt.pt \
        --attn-checkpoint checkpoints-attn-head/cbramod_ckpt.pt \
        --manifest test_manifest.csv --band sigma --scale-factors 0.5,0.75,1.0,1.25,1.5
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
    compute_pooled_scores,
    get_operating_threshold,
    load_model_checkpoint,
    perturb_window_band_power,
    resolve_pooling_config,
    seed_everything,
    setup_inference_cli_parser,
)
from p09c_clinical_subject_diagnostics import load_subject_ids_from_json
from p13_attention_mil_pooling import AttentionPoolingHead

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


def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Subject-Level Perturbation Test: p85 vs. Attention Pooling")
    group = parser.add_argument_group("Perturbation Test")
    group.add_argument("--band", type=str, default="sigma", choices=list(BAND_DEFS.keys()))
    group.add_argument("--scale-factors", type=str, default="0.5,0.75,1.0,1.25,1.5")
    group.add_argument("--filter-order", type=int, default=4)
    group.add_argument("--no-preserve-total-energy", dest="preserve_total_energy", action="store_false")
    group.add_argument("--max-windows-per-subject", type=int, default=None)
    group.add_argument(
        "--perturb-fraction", type=str, default="1.0",
        help="Comma-separated grid of fractions (each in (0, 1]) of a subject's windows to perturb "
             "(nested-prefix sampling) -- same semantics as p09i's flag of the same name."
    )
    group.add_argument("--subjects-json", type=str, default=None)
    group.add_argument("--output-csv", type=str, default="attention_pooled_perturbation.csv")

    attn_group = parser.add_argument_group("Attention Head")
    attn_group.add_argument("--attn-checkpoint", type=str, required=True, help="p13-trained attention head checkpoint")
    attn_group.add_argument("--attn-hidden-dim", type=int, default=64, help="Fallback only -- used if --attn-checkpoint predates saved architecture metadata")
    attn_group.add_argument("--attn-dropout", type=float, default=0.1, help="Inactive at eval() time; kept for AttentionPoolingHead's constructor signature")

    return parser.parse_args()


def main():
    args = parse_cli_args()
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./attention_pooled_perturbation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.features_pt:
        raise ValueError("This script perturbs the raw waveform directly (--manifest); --features-pt has no raw signal left to filter.")
    if args.num_classes != 2:
        raise ValueError("This script assumes binary classification (softmax[:, 1] as 'the' probability).")

    low, high = BAND_DEFS[args.band]
    scale_factors = np.array([float(s) for s in args.scale_factors.split(",")])
    lo_scale_idx, hi_scale_idx = int(np.argmin(scale_factors)), int(np.argmax(scale_factors))
    perturb_fractions = sorted(float(f) for f in args.perturb_fraction.split(","))
    for f in perturb_fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"--perturb-fraction values must be in (0, 1], got {f}.")

    print("Instantiating full CBraModE2EClassifier for raw waveform inference.")
    model = CBraModE2EClassifier(
        num_channels=args.num_channels, sfreq=args.sfreq, num_patches=args.num_patches,
        emb_dim=args.cbra_dim, hidden_dim=args.head_dim, num_classes=args.num_classes,
        head_type=args.head_type
    )
    model, ckpt_thresholds, _, ckpt_pooling_params = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    attn_ckpt = torch.load(args.attn_checkpoint, map_location="cpu", weights_only=True)
    if "attn_hidden_dim" in attn_ckpt:
        attn_hidden_dim = attn_ckpt["attn_hidden_dim"]
    else:
        attn_hidden_dim = args.attn_hidden_dim
        print(
            f"[Warning] --attn-checkpoint has no saved attn_hidden_dim metadata -- falling back to "
            f"--attn-hidden-dim ({attn_hidden_dim}); load_state_dict will fail below if that's wrong."
        )
    attn_head = AttentionPoolingHead(
        num_patches=args.num_patches, emb_dim=args.cbra_dim, hidden_dim=attn_hidden_dim, dropout=args.attn_dropout,
    ).to(device)
    attn_head.load_state_dict(attn_ckpt["attn_head_state_dict"])
    attn_head.eval()
    print(f"Loaded attention head from {args.attn_checkpoint} (epoch {attn_ckpt.get('epoch', '?')}).")

    pooling_strategy, top_percentile, t_window = resolve_pooling_config(
        pooling_strategy=args.pooling_strategy, top_percentile=args.top_percentile,
        t_window=args.t_window, ckpt_pooling_params=ckpt_pooling_params
    )
    if pooling_strategy == "all":
        raise ValueError("--pooling-strategy all doesn't apply here -- pick one specific pooling formula (this is the p85 side of the comparison).")
    threshold = get_operating_threshold(
        pooling_strategy=pooling_strategy, override_threshold=args.override_threshold, ckpt_thresholds=ckpt_thresholds
    )
    print(f"p85-side pooling strategy: {pooling_strategy} (top_percentile={top_percentile}, t_window={t_window}), operating threshold={threshold:.4f}")
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

    @torch.no_grad()
    def forward_both(batch_np: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Decomposes the fused CBraModE2EClassifier forward pass into backbone -> embedding -> head,
        since the attention head needs the intermediate embedding (not just the final window_prob)
        that p09i's single `model(x)` call never exposed. Returns (window_probs_np, feats, window_probs_t).
        """
        x = torch.from_numpy(batch_np).to(device)
        feats = model.backbone(x).mean(dim=1)  # [N, num_patches, emb_dim] -- same embedding p08a caches
        logits = model.head(feats)
        probs_t = torch.softmax(logits, dim=1)[:, 1]
        return probs_t.cpu().numpy(), feats, probs_t

    rows: List[Dict] = []
    for idx in tqdm(range(len(dataset)), desc="Process Subjects"):
        x_tensor, y_tensor, subj_id, stages, indices = dataset[idx]
        num_windows = x_tensor.shape[0]
        if num_windows == 0:
            continue

        raw_np = x_tensor.numpy()
        window_order = np.arange(num_windows)
        approximated = False
        if args.max_windows_per_subject and num_windows > args.max_windows_per_subject:
            window_order = rng.choice(num_windows, size=args.max_windows_per_subject, replace=False)
            approximated = True
        work_windows = raw_np[window_order]

        baseline_probs, baseline_feats, baseline_probs_t = forward_both(work_windows)
        baseline_p85 = compute_pooled_scores(baseline_probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window)
        with torch.no_grad():
            baseline_attn_prob, _ = attn_head(baseline_feats, baseline_probs_t)
        baseline_attn = float(baseline_attn_prob.item())

        permutation = rng.permutation(len(work_windows))

        for frac in perturb_fractions:
            n_perturb = max(1, int(round(frac * len(work_windows))))
            perturb_mask = np.zeros(len(work_windows), dtype=bool)
            perturb_mask[permutation[:n_perturb]] = True

            p85_per_scale = np.zeros(len(scale_factors))
            attn_per_scale = np.zeros(len(scale_factors))
            mean_window_prob_per_scale = np.zeros(len(scale_factors))
            for i, s in enumerate(scale_factors):
                perturbed_batch = np.stack([
                    perturb_window_band_power(
                        w, args.sfreq, low, high, s, order=args.filter_order,
                        preserve_total_energy=args.preserve_total_energy
                    ) if perturb_mask[j] else w
                    for j, w in enumerate(work_windows)
                ])
                probs, feats, probs_t = forward_both(perturbed_batch)
                p85_per_scale[i] = compute_pooled_scores(probs, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window)
                with torch.no_grad():
                    attn_prob, _ = attn_head(feats, probs_t)
                attn_per_scale[i] = float(attn_prob.item())
                mean_window_prob_per_scale[i] = probs.mean()

            p85_slope, p85_r2 = fit_local_slope(scale_factors, p85_per_scale)
            attn_slope, attn_r2 = fit_local_slope(scale_factors, attn_per_scale)

            mean_window_shift = mean_window_prob_per_scale[hi_scale_idx] - mean_window_prob_per_scale[lo_scale_idx]
            p85_shift = p85_per_scale[hi_scale_idx] - p85_per_scale[lo_scale_idx]
            attn_shift = attn_per_scale[hi_scale_idx] - attn_per_scale[lo_scale_idx]
            # Both ratios use the SAME denominator (identical underlying window_probs at every scale
            # factor) -- so any difference between them is attributable only to the pooling rule.
            p85_propagation_ratio = float(p85_shift / mean_window_shift) if abs(mean_window_shift) > 1e-9 else float("nan")
            attn_propagation_ratio = float(attn_shift / mean_window_shift) if abs(mean_window_shift) > 1e-9 else float("nan")

            rows.append({
                "subject_id": subj_id,
                "ground_truth": int(y_tensor.item()),
                "n_windows": len(work_windows),
                "perturb_fraction": frac,
                "n_windows_perturbed": int(n_perturb),
                "windows_approximated": approximated,
                "baseline_p85_score": float(baseline_p85),
                "baseline_attn_score": baseline_attn,
                "baseline_prediction_p85": int(baseline_p85 >= threshold),
                "p85_slope": p85_slope,
                "p85_r2": p85_r2,
                "p85_propagation_ratio": p85_propagation_ratio,
                "attn_slope": attn_slope,
                "attn_r2": attn_r2,
                "attn_propagation_ratio": attn_propagation_ratio,
                "mean_window_level_shift": float(mean_window_shift),
            })

    if not rows:
        print("No subjects processed -- nothing to report.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} (subject, perturb_fraction) rows to: {csv_path}")

    print("\n" + "=" * 88)
    print(f"DOES THE ATTENTION-POOLED SCORE PRESERVE THE SAME {args.band.upper()}-BAND CAUSAL EFFECT AS p85?")
    print("=" * 88)
    print(
        f"  {'metric':<28} {'p85':>12} {'attention':>12}"
    )
    print(
        f"  {'mean slope':<28} {df['p85_slope'].mean():+12.4f} {df['attn_slope'].mean():+12.4f}\n"
        f"  {'median slope':<28} {df['p85_slope'].median():+12.4f} {df['attn_slope'].median():+12.4f}\n"
        f"  {'frac(slope<0)':<28} {(df['p85_slope'] < 0).mean():12.2f} {(df['attn_slope'] < 0).mean():12.2f}\n"
        f"  {'mean R^2':<28} {df['p85_r2'].mean():12.3f} {df['attn_r2'].mean():12.3f}\n"
        f"  {'mean propagation_ratio':<28} {df['p85_propagation_ratio'].mean():+12.4f} {df['attn_propagation_ratio'].mean():+12.4f}\n"
        f"  {'median propagation_ratio':<28} {df['p85_propagation_ratio'].median():+12.4f} {df['attn_propagation_ratio'].median():+12.4f}"
    )

    valid = df["p85_propagation_ratio"].notna() & df["attn_propagation_ratio"].notna()
    paired_diff = df.loc[valid, "attn_propagation_ratio"] - df.loc[valid, "p85_propagation_ratio"]
    print(
        f"\n  Paired (attn - p85) propagation_ratio per (subject, fraction) row: "
        f"mean = {paired_diff.mean():+.4f}, median = {paired_diff.median():+.4f}, "
        f"frac(attn > p85) = {(paired_diff > 0).mean():.2f}  (n={int(valid.sum())})"
    )
    print(
        "\n  If attention pooling preserves the causal effect at least as well as p85, propagation_ratio "
        "should be comparable or higher for attention, and slope/frac(slope<0) should look similar --"
        " i.e. the sigma signal that lives in window_prob still reaches the final decision regardless "
        "of which pooling rule combines it. A much smaller attn_propagation_ratio (or attn_slope near "
        "zero while p85_slope is clearly negative) would mean attention pooling is NOT simply "
        "'inheriting' the causal effect the probe already carries -- e.g. if attention concentrates "
        "weight on windows where sigma's effect on window_prob happens to be weaker."
    )


if __name__ == "__main__":
    main()
