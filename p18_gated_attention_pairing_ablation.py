"""
p18_gated_attention_pairing_ablation.py

Direct test of a specific hypothesis about Option B's attn_weight/window_evidence relationship:
is the strong negative correlation between them (p17, within-subject median r=-0.49, 100% of
subjects) a functionally load-bearing mechanism the model actually relies on, or an arbitrary,
incidental byproduct of optimization that doesn't much affect performance either way? Correlating
either quantity against band power (p17) can't distinguish these -- it only shows neither is pure
noise, not whether the SPECIFIC pairing between them matters.

--------------------------------------------------------------------------
A flawed first design, caught before it ran, and why it's wrong
--------------------------------------------------------------------------
The obvious-looking approach -- for each subject, average many random within-subject shuffles of
window_evidence (keeping attn_weight fixed) into one "shuffled score" per subject, then compare
whole-cohort AUC/F1 of TRUE vs. UNIFORM vs. (averaged-)SHUFFLED -- is a mathematical tautology, not
an empirical test. Because attn_weight sums to 1 (softmax), E_permutation[sum(a_i * evidence_perm(i))]
= sum(a_i * mean(evidence)) = mean(evidence) EXACTLY, for ANY (a, evidence) pairing whatsoever --
regardless of whether the true pairing carries real information. Averaging many shuffles per subject
therefore converges to the UNIFORM baseline by linearity of expectation alone, with zero dependence
on whether the pairing matters. Verified this collapse happens in a synthetic sanity check before
discovering it was measuring nothing.

--------------------------------------------------------------------------
The correct design: a permutation test on the WHOLE-COHORT metric
--------------------------------------------------------------------------
Generate K independent "shuffled worlds": in world k, EVERY subject's window_evidence gets its own
independent random permutation (attn_weight stays put), producing ONE whole-cohort AUC/F1 for that
world -- not an average folded into a per-subject score. This gives an empirical null distribution
of "what population-level discrimination looks like under random (marginal-preserving) pairings."
Compare the model's ACTUAL (TRUE) whole-cohort AUC/F1 against that distribution: if TRUE sits far
outside it (e.g. above the 95th percentile), the specific learned pairing carries real, exploitable
class information beyond what the marginal distributions of attn_weight and window_evidence provide
alone -- validated on synthetic cases where the pairing is constructed to matter (TRUE lands at the
100th percentile of the shuffled distribution) vs. not (TRUE lands unremarkably within it, ~74th
percentile) before trusting this design.

Usage:
    python p18_gated_attention_pairing_ablation.py \
        --cache-dir /data/eeg_study/cache \
        --manifest /data/eeg_study/test_manifest.csv \
        --model-checkpoint checkpoints-gated-attn-linear/cbramod_ckpt.pt \
        --n-shuffles 300
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from cbramod_common import (
    CachedFeatureSubjectDataset,
    add_log_filename_argument,
    build_gated_attention_model,
    seed_everything,
    setup_cache_cli_parser,
    setup_common_cli_parser,
)
from p17_gated_attention_interpretability import compute_window_quantities
from cbramod_utils import setup_logger


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation test: does Option B's specific attn_weight/window_evidence pairing "
                    "carry real class information, or would any random (marginal-preserving) pairing "
                    "do about as well?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_common_cli_parser(parser)

    setup_cache_cli_parser(parser)

    ckpt_group = parser.add_argument_group("Checkpoint")
    ckpt_group.add_argument("--model-checkpoint", type=str, required=True)
    ckpt_group.add_argument("--attn-hidden-dim", type=int, default=32, help="Fallback only")
    ckpt_group.add_argument("--head-hidden-dim", type=int, default=64, help="Fallback only, irrelevant for a linear head")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--manifest", type=str, required=True)

    ablation_group = parser.add_argument_group("Permutation Test")
    ablation_group.add_argument(
        "--n-shuffles", type=int, default=300,
        help="Number of independent whole-cohort shuffled worlds (each with its own random "
             "per-subject permutation of window_evidence) making up the null distribution."
    )
    add_log_filename_argument(parser, __file__)

    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """Applies a FIXED threshold (never re-swept) -- same discipline as p13/p16's test-set scoring."""
    preds = (scores >= threshold).astype(int)
    return {
        "subject_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subject_accuracy": accuracy_score(labels, preds),
        "subject_sensitivity": recall_score(labels, preds),
        "subject_specificity": recall_score(labels, preds, pos_label=0),
        "roc_auc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5,
    }


def load_subject_ids(manifest_csv: str) -> List[str]:
    df = pd.read_csv(manifest_csv)
    return df["subject_id"].astype(str).tolist()


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = build_gated_attention_model(args, device, logger, require_head_type="linear")
    threshold = ckpt.get("optimal_threshold")
    if threshold is None:
        raise ValueError("--model-checkpoint has no saved optimal_threshold -- re-save it via p16, or pass one manually.")
    logger.info(f"threshold={threshold:.2f}")

    master_cache_path = Path(args.cache_dir) / args.master_cache_name
    dataset = CachedFeatureSubjectDataset(master_cache_path, filter_subject=load_subject_ids(args.manifest))
    logger.info(f"Running pairing permutation test on {len(dataset)} subjects ({args.n_shuffles} whole-cohort shuffled worlds)...")

    all_attn_weights, all_window_evidence, labels = [], [], []
    true_scores, uniform_scores = [], []
    max_reconstruction_error = 0.0
    for subj_idx in range(len(dataset)):
        bag_feats, label, _subject_id, _stages, _indices = dataset[subj_idx]
        attn_weights, window_evidence = compute_window_quantities(model, bag_feats, device)

        true_logit_diff = float(np.sum(attn_weights * window_evidence))

        # Sanity check: this MUST match the model's actual forward pass (same exact-decomposition
        # identity p17 validated numerically) -- if it doesn't, something upstream is inconsistent.
        with torch.no_grad():
            real_logits, _ = model(bag_feats.to(device).float())
        real_logit_diff = float((real_logits[1] - real_logits[0]).item())
        max_reconstruction_error = max(max_reconstruction_error, abs(true_logit_diff - real_logit_diff))

        all_attn_weights.append(attn_weights)
        all_window_evidence.append(window_evidence)
        true_scores.append(sigmoid(np.array([true_logit_diff]))[0])
        uniform_scores.append(sigmoid(np.array([float(np.mean(window_evidence))]))[0])
        labels.append(int(label.item()))

    logger.info(f"Max |reconstructed - actual| logit_diff across all subjects: {max_reconstruction_error:.2e} (should be ~float precision noise)")

    labels = np.array(labels)
    true_metrics = compute_metrics(np.array(true_scores), labels, threshold)
    uniform_metrics = compute_metrics(np.array(uniform_scores), labels, threshold)

    # The actual permutation test: K independent whole-cohort shuffled worlds, each with its own
    # per-subject random permutation of window_evidence (attn_weight untouched) -- NOT averaged
    # together, since that would collapse to uniform by construction (see module docstring).
    shuffled_aucs, shuffled_f1s = [], []
    for _ in range(args.n_shuffles):
        world_scores = [
            sigmoid(np.array([np.sum(aw * rng.permutation(we))]))[0]
            for aw, we in zip(all_attn_weights, all_window_evidence)
        ]
        world_metrics = compute_metrics(np.array(world_scores), labels, threshold)
        shuffled_aucs.append(world_metrics["roc_auc"])
        shuffled_f1s.append(world_metrics["subject_macro_f1"])
    shuffled_aucs, shuffled_f1s = np.array(shuffled_aucs), np.array(shuffled_f1s)

    print("\n" + "=" * 88)
    print("DOES THE SPECIFIC attn_weight / window_evidence PAIRING CARRY REAL CLASS INFORMATION?")
    print("=" * 88)
    print(f"{'condition':<28} {'F1':>8} {'AUC':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8}")
    print(
        f"{'TRUE (as trained)':<28} {true_metrics['subject_macro_f1']:8.4f} {true_metrics['roc_auc']:8.4f} "
        f"{true_metrics['subject_accuracy']:8.4f} {true_metrics['subject_sensitivity']:8.4f} {true_metrics['subject_specificity']:8.4f}"
    )
    print(
        f"{'UNIFORM weights':<28} {uniform_metrics['subject_macro_f1']:8.4f} {uniform_metrics['roc_auc']:8.4f} "
        f"{uniform_metrics['subject_accuracy']:8.4f} {uniform_metrics['subject_sensitivity']:8.4f} {uniform_metrics['subject_specificity']:8.4f}"
    )
    print(
        f"{'SHUFFLED null (mean±std)':<28} {shuffled_f1s.mean():8.4f} {shuffled_aucs.mean():8.4f} "
        f"{'':>8} {'':>8} {'':>8}   (std: F1={shuffled_f1s.std():.4f}, AUC={shuffled_aucs.std():.4f}, n={args.n_shuffles})"
    )

    auc_percentile = float((shuffled_aucs < true_metrics["roc_auc"]).mean())
    f1_percentile = float((shuffled_f1s < true_metrics["subject_macro_f1"]).mean())
    print(
        f"\n  TRUE AUC falls at the {auc_percentile:.0%} percentile of the shuffled null distribution "
        f"(TRUE F1: {f1_percentile:.0%} percentile)."
    )
    print(
        "\n  A high percentile (e.g. >=95%) means the specific learned pairing carries real, "
        "exploitable class information beyond what the marginal distributions of attn_weight and "
        "window_evidence provide alone -- the pairing is functionally load-bearing, not incidental. "
        "A middling percentile (TRUE indistinguishable from a typical random shuffle) would mean the "
        "model would do about as well with any random reassignment -- the strong correlation p17 "
        "found would then be more incidental than load-bearing. Either way, this does NOT by itself "
        "establish physical/semantic meaning for either quantity -- only whether their specific "
        "relationship is functionally necessary for the model's actual performance."
    )


if __name__ == "__main__":
    main()
