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
    CachedFeatureSubjectDataset,
    setup_common_cli_parser,
)
from p13_attention_mil_pooling import (
    AttentionPoolingHead,
    build_frozen_probe,
    frozen_window_probs,
    load_subject_ids,
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

    cache_group = parser.add_argument_group("Cache Controls")
    cache_group.add_argument("--cache-dir", type=str, required=True, help="Directory containing the master cache")
    cache_group.add_argument("--master-cache-name", type=str, default="cached_master_embeddings.pt")

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
    out_group.add_argument("--log-filename", type=str, default=Path(__file__).stem + ".log")

    return parser.parse_args()


# =====================================================================
# 2. CORRELATION REPORTING (same pattern as p09f/p09i/p09j/p09k/p09g)
# =====================================================================

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via plain rank + Pearson (no scipy.stats dependency needed)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def report_correlations(df: pd.DataFrame, reference_col: str, feature_cols: List[str], subject_col: str = "subject_id") -> None:
    """Pooled + within-subject Spearman correlation of `reference_col` against each of `feature_cols`."""
    print("\n" + "=" * 88)
    print(f"POOLED CORRELATION ({reference_col} vs. features, all windows/subjects together -- conflates within/between-subject variance)")
    print("=" * 88)
    for col in feature_cols:
        valid = df[col].notna()
        r = spearman_corr(df.loc[valid, reference_col].values, df.loc[valid, col].values)
        print(f"  {reference_col} vs {col:<24}: Spearman r = {r:+.4f}  (n={int(valid.sum())})")

    print("\n" + "=" * 88)
    print(f"WITHIN-SUBJECT CORRELATION ({reference_col} vs. features, summarized across subjects -- the direct test)")
    print("=" * 88)
    for col in feature_cols:
        per_subject_r = []
        for _, g in df.groupby(subject_col):
            valid = g[col].notna()
            if valid.sum() < 3:
                continue
            r = spearman_corr(g.loc[valid, reference_col].values, g.loc[valid, col].values)
            if not np.isnan(r):
                per_subject_r.append(r)
        per_subject_r = np.array(per_subject_r)
        if len(per_subject_r) == 0:
            print(f"  {reference_col} vs {col:<24}: no subjects had enough variance to compute this.")
            continue
        print(
            f"  {reference_col} vs {col:<24}: mean r = {per_subject_r.mean():+.4f}, "
            f"median r = {np.median(per_subject_r):+.4f}, "
            f"frac(r>0.2) = {(per_subject_r > 0.2).mean():.2f}, "
            f"frac(r<-0.2) = {(per_subject_r < -0.2).mean():.2f} (n_subjects={len(per_subject_r)})"
        )


# =====================================================================
# 3. MAIN
# =====================================================================

def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    probe = build_frozen_probe(args, device, logger)
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
    report_correlations(merged, "attn_weight", band_cols + yasa_cols)

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
    report_correlations(merged, "attn_weight", ["prob_extremity"])
    report_correlations(merged, "prob_extremity", band_cols + yasa_cols)


if __name__ == "__main__":
    main()
