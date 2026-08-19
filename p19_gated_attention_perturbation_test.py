"""
p19_gated_attention_perturbation_test.py

The single most important sanity check for Option B: does the WHOLE model (gated attention over
frozen embeddings + jointly-trained linear head, no separate probe at all) still show the same
sigma-band causal effect established for the original p85 pipeline (p09h/p09i) and confirmed to
survive Option A's pooling (p15)? If it doesn't, that's a much more fundamental problem than
anything the interpretability work (p17/p18) could reveal -- it would mean Option B's decision
doesn't rest on the same validated mechanism at all, regardless of what its internal attn_weight/
window_evidence quantities individually look like.

Mirrors p09i/p15's perturbation mechanics exactly (same raw-waveform band perturbation, scale-factor
sweep, --perturb-fraction nested-prefix design), but the backbone -> Option B pipeline decomposition
is simpler than p15's needed to be: Option B has no separate frozen probe to also compute a p85
comparison from. Every perturbed scale factor recomputes: raw window -> CBraModFeatureExtractor
(frozen backbone) -> fresh embedding -> GatedAttentionMIL (attention + linear head) -> subject-level
probability, directly -- exactly the real inference path, not an approximation of it.

Usage:
    python p19_gated_attention_perturbation_test.py \
        --model-checkpoint checkpoints-gated-attn-linear/cbramod_ckpt.pt \
        --manifest test_manifest.csv --data-dir /data/eeg_study/npy_files \
        --band sigma --scale-factors 0.5,0.75,1.0,1.25,1.5
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
    CBraModFeatureExtractor,
    PANSubjectEEGDataset,
    add_log_filename_argument,
    build_gated_attention_model,
    perturb_window_band_power,
    seed_everything,
    setup_perturbation_cli_parser,
    setup_common_cli_parser,
)
from cbramod_utils import setup_logger
from p09c_clinical_subject_diagnostics import load_subject_ids_from_json


def fit_local_slope(scale_factors: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    if len(scale_factors) < 2 or np.std(scale_factors) == 0:
        return float("nan"), float("nan")
    A = np.vstack([scale_factors, np.ones_like(scale_factors)]).T
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((values - pred) ** 2)
    ss_tot = np.sum((values - values.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def parse_cli_args() -> argparse.Namespace:
    # Deliberately setup_common_cli_parser, NOT setup_inference_cli_parser: the latter's --checkpoint
    # (required) and --pooling-strategy/--top-percentile/--t-window/--override-threshold all exist for
    # the OLD fused CBraModE2EClassifier + p85 pooling path (p09h/p09i/p15) -- Option B doesn't use any
    # of that, it loads its own model entirely via --model-checkpoint below, and pooling is whatever the
    # trained GatedAttentionMIL does, not a configurable p85-style choice.
    parser = argparse.ArgumentParser(
        description="Option B Subject-Level Perturbation Test (no separate probe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_common_cli_parser(parser)

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--manifest", type=str, required=True, help="Path to test_manifest.csv for raw .npy inference")
    data_group.add_argument("--subject-id", type=str, default=None, help="Optional comma-separated list of specific Subject IDs to analyze")
    data_group.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    setup_perturbation_cli_parser(parser, output_csv_default="gated_attention_perturbation.csv")
    perturb_group = parser.add_argument_group("Perturbation Test")
    perturb_group.add_argument(
        "--perturb-fraction", type=str, default="1.0",
        help="Comma-separated grid of fractions (each in (0, 1]) of a subject's windows to perturb "
             "(nested-prefix sampling) -- same semantics as p09i/p15's flag of the same name."
    )

    ckpt_group = parser.add_argument_group("Option B Model")
    ckpt_group.add_argument("--model-checkpoint", type=str, required=True)
    ckpt_group.add_argument("--attn-hidden-dim", type=int, default=32, help="Fallback only")
    ckpt_group.add_argument("--head-hidden-dim", type=int, default=64, help="Fallback only, irrelevant for a linear head")

    add_log_filename_argument(parser, __file__)

    return parser.parse_args()


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./gated_attention_perturbation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    low, high = BAND_DEFS[args.band]
    scale_factors = np.array([float(s) for s in args.scale_factors.split(",")])
    lo_scale_idx, hi_scale_idx = int(np.argmin(scale_factors)), int(np.argmax(scale_factors))
    perturb_fractions = sorted(float(f) for f in args.perturb_fraction.split(","))
    for f in perturb_fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"--perturb-fraction values must be in (0, 1], got {f}.")

    model, ckpt = build_gated_attention_model(args, device, logger)
    # Checked AFTER build_gated_attention_model(), not against the raw --num-classes flag: that flag
    # is only ever a cross-check/fallback (same convention as num_channels/sfreq below), so a stale/
    # default --num-classes 2 alongside an actual 3-class checkpoint would otherwise silently pass
    # this guard and proceed to compute a meaningless "binary" result from a 3-way output, instead of
    # loudly rejecting it as this script requires.
    if ckpt.get("num_classes", args.num_classes) != 2:
        raise ValueError("This script assumes binary classification.")
    threshold = ckpt.get("optimal_threshold", 0.5)
    print(f"Loaded Option B model from {args.model_checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"head_type={ckpt.get('head_type', 'mlp')}), threshold={threshold:.4f}")

    # num_channels/sfreq resolved from --model-checkpoint's own metadata (already cross-checked
    # against --num-channels/--sfreq, with a warning, inside build_gated_attention_model) -- this
    # backbone-only extractor is separate from the GatedAttentionMIL model, but must match the exact
    # backbone this checkpoint's embeddings were computed from.
    extractor = CBraModFeatureExtractor(
        num_channels=ckpt.get("num_channels", args.num_channels), sfreq=ckpt.get("sfreq", args.sfreq),
    ).to(device)
    extractor.eval()
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
    def forward_pooled(batch_np: np.ndarray) -> float:
        """raw perturbed windows -> frozen backbone embedding -> Option B's pooled subject-level probability."""
        x = torch.from_numpy(batch_np).to(device)
        feats = extractor(x)  # [N, num_patches, emb_dim] -- same embedding p08a caches
        logits, _attn_weights = model(feats)
        return float(torch.softmax(logits, dim=0)[1].item())

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

        baseline_prob = forward_pooled(work_windows)
        permutation = rng.permutation(len(work_windows))

        for frac in perturb_fractions:
            n_perturb = max(1, int(round(frac * len(work_windows))))
            perturb_mask = np.zeros(len(work_windows), dtype=bool)
            perturb_mask[permutation[:n_perturb]] = True

            pooled_per_scale = np.zeros(len(scale_factors))
            for i, s in enumerate(scale_factors):
                perturbed_batch = np.stack([
                    perturb_window_band_power(
                        w, args.sfreq, low, high, s, order=args.filter_order,
                        preserve_total_energy=args.preserve_total_energy
                    ) if perturb_mask[j] else w
                    for j, w in enumerate(work_windows)
                ])
                pooled_per_scale[i] = forward_pooled(perturbed_batch)

            slope, r2 = fit_local_slope(scale_factors, pooled_per_scale)
            pooled_shift = pooled_per_scale[hi_scale_idx] - pooled_per_scale[lo_scale_idx]

            rows.append({
                "subject_id": subj_id,
                "ground_truth": int(y_tensor.item()),
                "n_windows": len(work_windows),
                "perturb_fraction": frac,
                "n_windows_perturbed": int(n_perturb),
                "windows_approximated": approximated,
                "baseline_pooled_score": float(baseline_prob),
                "baseline_prediction": int(baseline_prob >= threshold),
                "subject_level_slope": slope,
                "subject_level_r2": r2,
                "pooled_score_shift": float(pooled_shift),
            })

    if not rows:
        print("No subjects processed -- nothing to report.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} (subject, perturb_fraction) rows to: {csv_path}")

    print("\n" + "=" * 88)
    print(f"DOES OPTION B (NO SEPARATE PROBE) SHOW THE SAME {args.band.upper()}-BAND CAUSAL EFFECT?")
    print("=" * 88)
    print(f"  mean subject-level slope   = {df['subject_level_slope'].mean():+.4f}")
    print(f"  median subject-level slope = {df['subject_level_slope'].median():+.4f}")
    print(f"  frac(slope<0) = {(df['subject_level_slope'] < 0).mean():.2f}   frac(slope>0) = {(df['subject_level_slope'] > 0).mean():.2f}")
    print(f"  mean subject-level fit R^2 = {df['subject_level_r2'].mean():.3f}")
    print(
        "\n  If Option B still relies on the same validated sigma mechanism, frac(slope<0) should be "
        "high (matching p85's ~0.95 and Option A's 1.00) and the sign should be NEGATIVE (more sigma "
        "power -> lower predicted probability), same direction as p85/Option A. If this comes back "
        "weak, inconsistent, or wrong-signed, that is a much more fundamental problem than anything "
        "p17/p18's interpretability work could reveal -- it would mean Option B's decision doesn't "
        "rest on the same validated mechanism at all, regardless of what its internal attn_weight/ "
        "window_evidence quantities look like in isolation."
    )


if __name__ == "__main__":
    main()
