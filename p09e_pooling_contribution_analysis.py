"""
p09e_pooling_contribution_analysis.py

Answers a question that's logically prior to (and cheaper than) feature
attribution or morphology tagging: "given the pooling formula actually in
use, which windows does the math itself say determine this subject's
pooled score?"

Subject-level scores here are produced by `compute_pooled_scores()` --
p85_score (percentile), top_10_mean, trimmed_top_10, or burden_ratio --
applied to a vector of per-window probabilities. None of these are black
boxes: they're simple, well-defined, low-dimensional functions of that
vector, so "which windows matter" has an exact, retraining-free answer via
leave-one-out (LOO): for each window, remove it, recompute the pooled score,
and see how much it moved. See `compute_leave_one_out_contributions()` in
cbramod_common.py for the implementation and cost analysis (O(N^2), cheap
since N = windows per subject is small).

This deliberately does NOT try to explain *why* a contributing window's own
classifier output scored the way it did -- that's a separate question
(feature attribution / morphology characterization, see
p10_feature_attribution.py), and should be pointed at whichever windows
*this* script identifies as actually mattering, rather than an arbitrary
tier (e.g. p09c's "Tier 1: Top Drivers" is the right target for
p85_score/top_10_mean/trimmed_top_10's upper-tail-driven contributors, but
NOT for burden_ratio, where every window crossing the per-window threshold
contributes equally -- not particularly the single highest-probability one).

Different pooling methods have qualitatively different contribution
"shapes", worth knowing before interpreting the ranked list:
  - p85_score (percentile, linearly interpolated): contribution is smeared
    across the few windows adjacent to the 85th-percentile rank, not
    concentrated in one sharp window.
  - top_10_mean / trimmed_top_10: contribution is exactly zero for every
    window outside the top-k set, nonzero only within it.
  - burden_ratio: a clean two-valued split -- every window at/above
    --t-window gets one (positive) contribution value, every window below
    it gets another (negative) value; nothing in between.

Usage:
  python p09e_pooling_contribution_analysis.py \
      --checkpoint model.pt --manifest test_manifest.csv \
      --subject-id GRINS0322 --top-k-contributors 5 --output-dir results/
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cbramod_common import (
    CBraModE2EClassifier,
    CachedFeatureSubjectDataset,
    LinearProbeHead,
    MLPProbeHead,
    PANSubjectEEGDataset,
    compute_leave_one_out_contributions,
    load_model_checkpoint,
    resolve_pooling_config,
    setup_inference_cli_parser
)
from cbramod_utils import seed_everything
from p09c_clinical_subject_diagnostics import SubjectEEGInspector, load_subject_ids_from_json


def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Leave-One-Out Pooling Contribution Analysis")
    contrib_group = parser.add_argument_group("Pooling Contribution Analysis")
    contrib_group.add_argument(
        "--top-k-contributors", type=int, default=5,
        help="Number of highest-|contribution| windows to print and export per subject."
    )
    contrib_group.add_argument(
        "--subjects-json", type=str, default=None,
        help="Path to a p09d_subject_confidence_report.py --output-json report. Its subject_ids are "
             "unioned with --subject-id (if also given) to select which subjects to analyze."
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./pooling_contribution_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    window_sec = float(args.num_patches)

    # 1. Instantiate Model Architecture (same branching as p09c/p09/p10)
    if args.features_pt:
        print("Instantiating isolated Probe Head for cached feature inference.")
        if args.head_type == "linear":
            model = LinearProbeHead(num_patches=args.num_patches, emb_dim=args.cbra_dim, num_classes=args.num_classes)
        else:
            model = MLPProbeHead(
                num_patches=args.num_patches, emb_dim=args.cbra_dim, hidden_dim=args.head_dim,
                num_classes=args.num_classes, dropout=args.dropout
            )
    else:
        print("Instantiating full CBraModE2EClassifier for raw waveform inference.")
        model = CBraModE2EClassifier(
            num_channels=args.num_channels, sfreq=args.sfreq, num_patches=args.num_patches,
            emb_dim=args.cbra_dim, hidden_dim=args.head_dim, num_classes=args.num_classes,
            head_type=args.head_type
        )

    # 2. Load Model Checkpoint
    model, _, _, ckpt_pooling_params = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    # 2b. Resolve Pooling Config: CLI override > checkpoint's training-time config > hardcoded default
    pooling_strategy, top_percentile, t_window = resolve_pooling_config(
        pooling_strategy=args.pooling_strategy,
        top_percentile=args.top_percentile,
        t_window=args.t_window,
        ckpt_pooling_params=ckpt_pooling_params
    )
    print(
        f"Pooling config: strategy={pooling_strategy}, top_percentile={top_percentile}, t_window={t_window} "
        f"(source: {'CLI override' if args.pooling_strategy is not None else 'checkpoint/default'})"
    )
    if pooling_strategy == "all":
        raise ValueError(
            "--pooling-strategy all doesn't apply here -- leave-one-out contribution analysis needs one "
            "specific pooling formula to differentiate with respect to. Pick a single strategy."
        )

    # SubjectEEGInspector's own threshold only affects report['prediction'] (context for the export), not
    # the contribution computation itself, which depends only on pooling_strategy/top_percentile/t_window.
    inspector = SubjectEEGInspector(model=model, device=device, threshold=0.5)

    # 2c. Resolve subject filter: union of --subject-id and --subjects-json's subject_ids
    subject_filter = [s.strip() for s in args.subject_id.split(",")] if args.subject_id else []
    if args.subjects_json:
        json_subject_ids = load_subject_ids_from_json(Path(args.subjects_json))
        print(f"Loaded {len(json_subject_ids)} subject_id(s) from --subjects-json: {args.subjects_json}")
        subject_filter = subject_filter + [sid for sid in json_subject_ids if sid not in subject_filter]
    subject_filter = subject_filter or None

    # 3. Load dataset
    if args.features_pt:
        dataset = CachedFeatureSubjectDataset(args.features_pt, filter_subject=subject_filter)
        print(f"Loaded cached features for {len(dataset)} subjects.")
    else:
        dataset = PANSubjectEEGDataset(
            manifest_csv=args.manifest, data_dir=args.data_dir, filter_stage=args.filter_stage,
            filter_subject=subject_filter, memory_map=True
        )
        print(f"Loaded raw EEG recording dataset for {len(dataset)} subjects.")

    # 4. Process subjects: compute leave-one-out contributions, print & export JSON
    tier_name = f"Pooling Contributors ({pooling_strategy})"
    for idx in tqdm(range(len(dataset)), desc="Process Subjects"):
        x_tensor, y_tensor, subj_id, stages, indices = dataset[idx]
        if x_tensor.shape[0] == 0:
            print(f"Skipping {subj_id}: No valid windows after stage filtering.")
            continue

        report = inspector.inspect_subject(
            x_tensor=x_tensor,
            subject_id=subj_id,
            ground_truth=y_tensor.item(),
            stages=stages,
            indices=indices,
            pooling_strategy=pooling_strategy,
            top_percentile=top_percentile,
            t_window=t_window,
            batch_size=args.batch_size
        )

        contributions = compute_leave_one_out_contributions(
            report["window_probs"], method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )
        ranked_order = np.argsort(np.abs(contributions))[::-1][:args.top_k_contributors]

        print(
            f"\n=== {subj_id} | GT={report['ground_truth']} | pooled_score={report['pooled_score']:.4f} "
            f"| {len(report['window_probs'])} windows ==="
        )
        contributor_records: List[dict] = []
        for f_idx in ranked_order:
            record = SubjectEEGInspector._window_record(
                int(f_idx), report["window_probs"], report["indices"], report["stages"], window_sec
            )
            record["contribution"] = float(contributions[f_idx])
            contributor_records.append(record)
            print(
                f"  raw_epoch_index={record['raw_epoch_index']:>4d} @ {record['start_time_sec']:.0f}s "
                f"(stage={record['stage']}) probability={record['probability']:.4f} "
                f"contribution={record['contribution']:+.4f}"
            )

        json_path = output_dir / f"{subj_id}_pooling_contributors.json"
        export_payload = {
            "subject_id": subj_id,
            "ground_truth": report["ground_truth"],
            "prediction": report["prediction"],
            "pooled_score": float(report["pooled_score"]),
            "pooling_strategy": pooling_strategy,
            "top_percentile": top_percentile,
            "t_window": t_window,
            "priority_tiers": {tier_name: contributor_records}
        }
        with open(json_path, "w") as f:
            json.dump(export_payload, f, indent=4)
        print(f"-> Saved pooling contributor windows: {json_path}")


if __name__ == "__main__":
    main()
