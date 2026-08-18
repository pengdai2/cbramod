"""
p14_attention_weight_interpretability.py

Interpretability follow-up to p13_attention_mil_pooling.py: does the learned attention gate weight
windows in a way that echoes the sigma-band causal story from the earlier investigation
(docs/sigma_band_causal_investigation.md), or is the ~0.036 AUC gain over p85 coming from something
opaque/spurious?

Dumps the per-window attention weight the trained gate assigns to every window of every subject in
--manifest, joins that against p09k_absolute_band_power_analysis.py's per-window morphology CSV
(subject_id + raw_epoch_index/window_idx is the join key -- both ultimately trace back to the same
p02 window_idx numbering), and reports the same pooled/within-subject Spearman correlation structure
used throughout the causal investigation: attention weight vs. absolute band power (delta/theta/
alpha/sigma/beta) and vs. YASA spindle/slow-wave counts.

Usage:
    python p14_attention_weight_interpretability.py \
        --cache-dir /data/eeg_study/cache \
        --manifest /data/eeg_study/test_manifest.csv \
        --probe-checkpoint /data/eeg_study/checkpoints-probe-linear/cbramod_ckpt.pt \
        --attn-checkpoint /data/eeg_study/checkpoints-attn-head/cbramod_ckpt.pt \
        --morphology-csv /data/eeg_study/analysis/test_ckpt/absolute_band_power_analysis.csv \
        --output-csv attention_weight_interpretability.csv
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from cbramod_common import (
    AttentionPoolingHead,
    CachedFeatureSubjectDataset,
    add_log_filename_argument,
    build_frozen_probe,
    frozen_window_probs,
    load_subject_ids,
    report_reference_correlations,
    setup_cache_cli_parser,
    setup_common_cli_parser,
    spearman_corr,
)
from cbramod_utils import setup_logger


# =====================================================================
# 1. CLI
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dumps per-window attention weights from a trained p13 attention head and "
                    "correlates them against band power / YASA event counts (p09k's output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_common_cli_parser(parser)

    setup_cache_cli_parser(parser)

    ckpt_group = parser.add_argument_group("Checkpoints")
    ckpt_group.add_argument("--probe-checkpoint", type=str, required=True, help="p08b-trained probe head checkpoint")
    ckpt_group.add_argument("--attn-checkpoint", type=str, required=True, help="p13-trained attention head checkpoint")
    ckpt_group.add_argument("--attn-hidden-dim", type=int, default=64, help="Fallback only -- used if --attn-checkpoint predates saved architecture metadata")
    ckpt_group.add_argument("--attn-dropout", type=float, default=0.1, help="Inactive at eval() time; kept for AttentionPoolingHead's constructor signature")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--manifest", type=str, required=True, help="Subject manifest CSV (e.g. p03's test_manifest.csv)")
    data_group.add_argument(
        "--morphology-csv", type=str, required=True,
        help="p09k_absolute_band_power_analysis.py's output CSV for the SAME subjects -- must have "
             "been run on the same window set (same manifest/stage filter) for the join to line up."
    )

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-csv", type=str, default="attention_weight_interpretability.csv")
    add_log_filename_argument(parser, __file__)

    return parser.parse_args()


# =====================================================================
# 2. CORRELATION REPORTING (same pattern as p09f/p09i/p09j/p09k/p09g)
# =====================================================================

def partial_spearman_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial Spearman correlation of x and y, controlling for z, via the standard partial-correlation
    formula applied to rank-transformed values: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)).
    Answers a different, stronger question than just eyeballing whether r_xz and r_yz are both large --
    it directly measures whether the x-y relationship survives once z's shared influence on both is
    removed, rather than requiring a napkin-math guess at how much of r_xy a mediating variable could
    plausibly account for.
    """
    r_xy, r_xz, r_yz = spearman_corr(x, y), spearman_corr(x, z), spearman_corr(y, z)
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def report_partial_correlation(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, subject_col: str = "subject_id"
) -> None:
    """Raw vs. partial (controlling for z_col) Spearman correlation of x_col and y_col, pooled + within-subject."""
    valid = df[[x_col, y_col, z_col]].notna().all(axis=1)
    r_raw_pooled = spearman_corr(df.loc[valid, x_col].values, df.loc[valid, y_col].values)
    r_partial_pooled = partial_spearman_corr(
        df.loc[valid, x_col].values, df.loc[valid, y_col].values, df.loc[valid, z_col].values
    )
    print(
        f"  POOLED: raw Spearman({x_col}, {y_col}) = {r_raw_pooled:+.4f}  |  "
        f"partial (controlling for {z_col}) = {r_partial_pooled:+.4f}  (n={int(valid.sum())})"
    )

    raw_rs, partial_rs = [], []
    for _, g in df.groupby(subject_col):
        gvalid = g[[x_col, y_col, z_col]].notna().all(axis=1)
        if gvalid.sum() < 4:
            continue
        r_raw = spearman_corr(g.loc[gvalid, x_col].values, g.loc[gvalid, y_col].values)
        r_partial = partial_spearman_corr(
            g.loc[gvalid, x_col].values, g.loc[gvalid, y_col].values, g.loc[gvalid, z_col].values
        )
        if not np.isnan(r_raw):
            raw_rs.append(r_raw)
        if not np.isnan(r_partial):
            partial_rs.append(r_partial)
    raw_rs, partial_rs = np.array(raw_rs), np.array(partial_rs)
    if len(raw_rs) == 0 or len(partial_rs) == 0:
        print(f"  WITHIN-SUBJECT: not enough per-subject variance to compute this.")
        return
    print(
        f"  WITHIN-SUBJECT: raw median r = {np.median(raw_rs):+.4f} (n_subjects={len(raw_rs)})  |  "
        f"partial median r (controlling for {z_col}) = {np.median(partial_rs):+.4f} (n_subjects={len(partial_rs)})"
    )
    print(
        f"  If the partial median stays close to the raw median, {z_col} explains little of the "
        f"{x_col}-{y_col} relationship. If it collapses toward 0, {z_col} was doing most of the work."
    )


# =====================================================================
# 3. MAIN
# =====================================================================

def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    probe, _probe_ckpt = build_frozen_probe(args, device, logger)
    logger.info(f"Loaded frozen probe from {args.probe_checkpoint}")

    attn_ckpt = torch.load(args.attn_checkpoint, map_location="cpu", weights_only=True)
    if "attn_hidden_dim" in attn_ckpt:
        hidden_dim = attn_ckpt["attn_hidden_dim"]
    else:
        hidden_dim = args.attn_hidden_dim
        logger.warning(
            f"--attn-checkpoint has no saved attn_hidden_dim metadata -- falling back to "
            f"--attn-hidden-dim ({hidden_dim}); load_state_dict will fail below if that's wrong."
        )
    attn_head = AttentionPoolingHead(
        num_patches=args.num_patches, emb_dim=args.cbra_dim, hidden_dim=hidden_dim, dropout=args.attn_dropout,
    ).to(device)
    attn_head.load_state_dict(attn_ckpt["attn_head_state_dict"])
    attn_head.eval()
    logger.info(f"Loaded attention head from {args.attn_checkpoint} (epoch {attn_ckpt.get('epoch', '?')})")

    subject_ids = load_subject_ids(args.manifest)
    dataset = CachedFeatureSubjectDataset(master_cache_path, filter_subject=subject_ids)
    logger.info(f"Dumping attention weights for {len(dataset)} subjects...")

    rows = []
    with torch.no_grad():
        for subj_idx in range(len(dataset)):
            bag_feats, _label, subject_id, _stages, indices = dataset[subj_idx]
            bag_feats = bag_feats.to(device).float()

            window_probs = frozen_window_probs(probe, bag_feats, device)
            _subject_prob, attn_weights = attn_head(bag_feats, window_probs)

            attn_weights_np = attn_weights.cpu().numpy()
            window_probs_np = window_probs.cpu().numpy()
            for i, window_idx in enumerate(indices):
                rows.append({
                    "subject_id": subject_id,
                    "raw_epoch_index": int(window_idx),
                    "attn_weight": float(attn_weights_np[i]),
                    "window_prob": float(window_probs_np[i]),
                })

    attn_df = pd.DataFrame(rows)
    logger.info(f"Collected {len(attn_df)} per-window attention weights across {attn_df['subject_id'].nunique()} subjects.")

    morph_df = pd.read_csv(args.morphology_csv)
    morph_df["subject_id"] = morph_df["subject_id"].astype(str)
    attn_df["subject_id"] = attn_df["subject_id"].astype(str)

    merged = attn_df.merge(morph_df, on=["subject_id", "raw_epoch_index"], how="inner")
    logger.info(
        f"Joined against {args.morphology_csv}: {len(merged)}/{len(attn_df)} attention-weight rows "
        f"matched a morphology row (subject_id, raw_epoch_index). Unmatched rows are dropped -- most "
        f"likely stage-filtering differences between this manifest's window set and the one "
        f"p09k was run with; re-run p09k on the same manifest/stage filter if this drop is large."
    )
    if merged.empty:
        logger.error("No rows survived the join -- nothing to correlate. Check --morphology-csv matches --manifest.")
        return

    merged["prob_extremity"] = (merged["window_prob"] - 0.5).abs()

    output_path = Path(args.output_csv)
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved {len(merged)} joined rows to {output_path}")

    band_cols = [c for c in merged.columns if c.endswith("_real_abspower")]
    yasa_cols = [c for c in ("n_spindles", "n_slow_waves") if c in merged.columns]

    print(
        "\nDoes the learned attention gate echo the sigma-band causal story (elevated sigma power -> "
        "lower predicted probability), or key on something else entirely? A negative attn_weight vs. "
        "sigma_real_abspower correlation here would mean the gate DOWNWEIGHTS high-sigma-power windows -- "
        "consistent with the model treating them as evidence against the patient class and the gate "
        "learning to suppress their influence on the pooled score, though the causal investigation's "
        "perturbation tests (not a correlation like this one) remain the actual causal evidence."
    )
    report_reference_correlations(merged, "attn_weight", band_cols + yasa_cols)

    print(
        "\n" + "=" * 88
        + "\nEXTREMITY CHECK: is the gate filtering by how INFORMATIVE the frozen probe's own prediction "
          "is for a window, rather than (or in addition to) discovering a new spectral marker?\n"
        + "=" * 88
        + "\nprob_extremity = |window_prob - 0.5| -- how far the frozen probe's own window-level "
          "prediction sits from maximally uninformative. If attn_weight correlates POSITIVELY with "
          "prob_extremity, the gate favors windows where the probe already has something confident to "
          "say, regardless of which band drives that. If delta_real_abspower correlates NEGATIVELY "
          "with prob_extremity too, that's consistent with high-delta windows being less discriminative "
          "for the probe in the first place -- i.e. the gate's strong delta downweighting could be "
          "explained as 'downweight uninformative windows' rather than 'delta itself is the signal'."
    )
    report_reference_correlations(merged, "attn_weight", ["prob_extremity"])
    report_reference_correlations(merged, "prob_extremity", band_cols + yasa_cols)

    print(
        "\n" + "=" * 88
        + "\nSTAGE COMPOSITION CHECK: is the delta downweighting a genuine within-window-content effect, "
          "or mostly a proxy for N2-vs-N3 stage composition?\n"
        + "=" * 88
        + "\nThe attn_weight vs delta_real_abspower correlation (-0.81 within-subject median, pooled "
          "across N2+N3) could reflect the gate genuinely reacting to delta content window-by-window, "
          "OR it could be dominated by delta differing systematically BETWEEN stages (N3 windows "
          "having far more delta than N2) with the gate really just downweighting N3-as-a-category -- "
          "in which case the correlation should weaken substantially once computed WITHIN a single "
          "stage, where delta variance is only ever within-stage, never between-stage."
    )
    if "stage" not in merged.columns:
        print("  No 'stage' column in the joined data -- skipping.")
    else:
        stage_summary = merged.groupby("stage").agg(
            n_windows=("attn_weight", "count"),
            n_subjects=("subject_id", "nunique"),
            mean_attn_weight=("attn_weight", "mean"),
            mean_delta_abspower=("delta_real_abspower", "mean"),
        )
        print("\n--- Stage composition (does attn_weight/delta already differ BETWEEN stages?) ---")
        print(stage_summary.to_string(float_format=lambda x: f"{x:.4f}"))

        for stage_name, group in merged.groupby("stage"):
            if len(group) < 50:
                print(f"\n--- WITHIN STAGE = {stage_name} -- skipped, only {len(group)} windows ---")
                continue
            print(f"\n--- WITHIN STAGE = {stage_name} ONLY ({len(group)} windows, {group['subject_id'].nunique()} subjects) ---")
            report_reference_correlations(group, "attn_weight", band_cols + yasa_cols)

    print(
        "\n" + "=" * 88
        + "\nTIME-OF-NIGHT CHECK: is the delta downweighting actually a position-in-recording confound?\n"
        + "=" * 88
        + "\nDelta/slow-wave power is well known to decline across the night as sleep pressure "
          "dissipates. If the gate simply learned 'later windows in the recording are more "
          "trustworthy' for some reason UNRELATED to delta content itself, and delta happens to "
          "decline over that same span, that alone would produce a strong attn_weight-vs-delta "
          "correlation without the gate actually reacting to delta. raw_epoch_index (this subject's "
          "own window ordering, a proxy for time-of-night/position-in-recording) lets us check both "
          "halves of that chain directly."
    )
    report_reference_correlations(merged, "raw_epoch_index", ["attn_weight", "delta_real_abspower"])

    print(
        "\n--- Does attn_weight vs delta_real_abspower SURVIVE controlling for raw_epoch_index? ---\n"
        "A napkin-math guess (multiplying the two legs' correlation magnitudes above) isn't a real "
        "test of mediation -- this partial correlation is: it removes whatever of the attn_weight-delta "
        "relationship raw_epoch_index could explain, then reports what's left."
    )
    report_partial_correlation(merged, "attn_weight", "delta_real_abspower", "raw_epoch_index")


if __name__ == "__main__":
    main()
