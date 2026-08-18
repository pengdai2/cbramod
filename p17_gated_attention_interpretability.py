"""
p17_gated_attention_interpretability.py

Interpretability follow-up to Option B (p16_gated_attention_embedding_mil.py), analogous to what
p14 did for Option A -- but with a genuinely cleaner quantity to interpret, not just a proxy.

--------------------------------------------------------------------------
Why a per-window "evidence" term here is EXACT, not a proxy
--------------------------------------------------------------------------
Because pooled = sum(attn_weight_i * flat_i) and (with --head-type linear) the head is a single
nn.Linear, logit = head(pooled) decomposes EXACTLY:

    head(pooled) = W @ sum(a_i * flat_i) + b = sum(a_i * (W @ flat_i)) + b
                 = sum(a_i * (W @ flat_i + b))   [since sum(a_i) = 1, softmax attn_weights]
                 = sum(a_i * head(flat_i))

So `head(flat_i)` -- the SAME linear head applied to each window's own (pre-pooling) embedding --
is not an approximation of that window's contribution to the final decision. It IS that
contribution, up to the attn_weight multiplication. This only holds for --head-type linear; an MLP
head does not commute with the weighted sum (MLP(sum(a_i x_i)) != sum(a_i MLP(x_i)) in general), so
this script requires a linear-head checkpoint and refuses to run against an MLP one rather than
silently reporting a quantity that no longer means what its docstring claims.

Reports the same pooled/within-subject Spearman correlation structure used throughout the causal
investigation, for THREE questions:
  1. attn_weight vs. band power / YASA counts (same question p14 asked for Option A).
  2. window_evidence vs. band power / YASA counts (NEW: what does each window's own, attention-
     independent "vote" say about its content -- the closest analog to Option A's window_prob,
     except this one comes from a head trained jointly with the gate, not inherited from the old
     probe).
  3. attn_weight vs. window_evidence (does the gate preferentially amplify windows its own head
     already finds informative/extreme -- an EXACT version of the "informativeness filtering"
     hypothesis p14 could only test approximately via prob_extremity, since window_evidence here is
     the model's own intrinsic per-window signal, not an external proxy).

Usage:
    python p17_gated_attention_interpretability.py \
        --cache-dir /data/eeg_study/cache \
        --manifest /data/eeg_study/test_manifest.csv \
        --model-checkpoint checkpoints-gated-attn/cbramod_ckpt.pt \
        --morphology-csv /data/eeg_study/analysis/test_ckpt/absolute_band_power_analysis.csv
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from cbramod_common import (
    CachedFeatureSubjectDataset,
    GatedAttentionMIL,
    add_log_filename_argument,
    build_gated_attention_model,
    load_subject_ids,
    report_reference_correlations,
    setup_cache_cli_parser,
    setup_common_cli_parser,
)
from cbramod_utils import setup_logger


# =====================================================================
# 1. CLI
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dumps per-window attn_weight and (linear-head-only) window_evidence from a "
                    "trained p16 Option B model, correlated against band power / YASA event counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_common_cli_parser(parser)

    setup_cache_cli_parser(parser)

    ckpt_group = parser.add_argument_group("Checkpoint")
    ckpt_group.add_argument("--model-checkpoint", type=str, required=True, help="p16-trained Option B checkpoint")
    ckpt_group.add_argument("--attn-hidden-dim", type=int, default=32, help="Fallback only -- used if --model-checkpoint predates saved architecture metadata")
    ckpt_group.add_argument("--head-hidden-dim", type=int, default=64, help="Fallback only, and irrelevant for a linear head")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--manifest", type=str, required=True)
    data_group.add_argument("--morphology-csv", type=str, required=True, help="p09k_absolute_band_power_analysis.py's output CSV for the SAME subjects")

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-csv", type=str, default="gated_attention_interpretability.csv")
    add_log_filename_argument(parser, __file__)

    return parser.parse_args()


# =====================================================================
# 2. PER-WINDOW QUANTITY EXTRACTION
# =====================================================================

@torch.no_grad()
def compute_window_quantities(model: GatedAttentionMIL, bag_feats: torch.Tensor, device: torch.device):
    """
    Recomputes GatedAttentionMIL.forward()'s internals to ALSO expose the per-window evidence term
    (the head applied to each window's own embedding, before pooling) -- p16's own forward() only
    returns the pooled logits + attn_weights, not this intermediate, since training never needs it.
    """
    bag_feats = bag_feats.to(device).float()
    n_windows = bag_feats.shape[0]
    flat = model.norm(bag_feats.reshape(n_windows, -1))
    gated = torch.tanh(model.V(flat)) * torch.sigmoid(model.U(flat))
    scores = model.w(model.gate_dropout(gated)).squeeze(-1)
    attn_weights = torch.softmax(scores, dim=0)
    per_window_logits = model.head(flat)  # [n_windows, num_classes] -- head applied BEFORE pooling
    window_evidence = per_window_logits[:, 1] - per_window_logits[:, 0]
    return attn_weights.cpu().numpy(), window_evidence.cpu().numpy()


# =====================================================================
# 3. MAIN
# =====================================================================

def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ckpt = build_gated_attention_model(args, device, logger, require_head_type="linear")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    subject_ids = load_subject_ids(args.manifest)
    dataset = CachedFeatureSubjectDataset(master_cache_path, filter_subject=subject_ids)
    logger.info(f"Dumping attn_weight/window_evidence for {len(dataset)} subjects...")

    rows = []
    for subj_idx in range(len(dataset)):
        bag_feats, _label, subject_id, _stages, indices = dataset[subj_idx]
        attn_weights, window_evidence = compute_window_quantities(model, bag_feats, device)
        for i, window_idx in enumerate(indices):
            rows.append({
                "subject_id": subject_id,
                "raw_epoch_index": int(window_idx),
                "attn_weight": float(attn_weights[i]),
                "window_evidence": float(window_evidence[i]),
            })

    attn_df = pd.DataFrame(rows)
    logger.info(f"Collected {len(attn_df)} per-window rows across {attn_df['subject_id'].nunique()} subjects.")

    morph_df = pd.read_csv(args.morphology_csv)
    morph_df["subject_id"] = morph_df["subject_id"].astype(str)
    attn_df["subject_id"] = attn_df["subject_id"].astype(str)
    merged = attn_df.merge(morph_df, on=["subject_id", "raw_epoch_index"], how="inner")
    logger.info(
        f"Joined against {args.morphology_csv}: {len(merged)}/{len(attn_df)} rows matched "
        f"(subject_id, raw_epoch_index). A large drop usually means --morphology-csv is stale "
        f"relative to the current master cache (e.g. predates a reslice) -- re-run p09k fresh if so."
    )
    if merged.empty:
        logger.error("No rows survived the join -- nothing to correlate.")
        return

    output_path = Path(args.output_csv)
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved {len(merged)} joined rows to {output_path}")

    band_cols = [c for c in merged.columns if c.endswith("_real_abspower")]
    yasa_cols = [c for c in ("n_spindles", "n_slow_waves") if c in merged.columns]

    print(
        "\nQuestion 1: does the gate's attention weighting echo the same content axis Option A's did "
        "(delta down, beta/spindle up), or something different, now that there's no separate frozen "
        "probe for a causal signal to hide behind?"
    )
    report_reference_correlations(merged, "attn_weight", band_cols + yasa_cols)

    print(
        "\nQuestion 2: what does each window's OWN vote (window_evidence -- exact, not a proxy, per "
        "this file's docstring) say about its spectral/morphological content? This is the closest "
        "analog to Option A's window_prob, except it comes from a head trained jointly with the "
        "gate here, not inherited from the old probe."
    )
    report_reference_correlations(merged, "window_evidence", band_cols + yasa_cols)

    print(
        "\nQuestion 3: does the gate preferentially amplify windows its own head already finds "
        "informative/extreme? An EXACT version of the informativeness-filtering hypothesis (p14 could "
        "only test this approximately, via an external prob_extremity proxy computed from a DIFFERENT "
        "model's output). A positive correlation here means attn_weight tracks |window_evidence| or "
        "window_evidence's own extremity -- check the sign/magnitude of both to distinguish 'gate "
        "trusts confident windows' from 'gate trusts windows favoring one particular class'."
    )
    merged["window_evidence_abs"] = merged["window_evidence"].abs()
    report_reference_correlations(merged, "attn_weight", ["window_evidence", "window_evidence_abs"])


if __name__ == "__main__":
    main()
