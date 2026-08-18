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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cbramod_common import (
    CachedFeatureSubjectDataset,
    PANSubjectEEGDataset,
    add_log_filename_argument,
    build_e2e_classifier,
    build_frozen_probe,
    compute_leave_one_out_contributions,
    extract_ckpt_metadata,
    get_operating_threshold,
    resolve_pooling_config,
    seed_everything,
    setup_inference_cli_parser,
)
from cbramod_utils import setup_logger
from p09c_clinical_subject_diagnostics import SubjectEEGInspector, load_subject_ids_from_json


# -----------------------------------------------------------------------------
# Temporal clustering null-model test
# -----------------------------------------------------------------------------
#
# Contributor windows can look "clustered in time" purely because of the
# stage-filtering already applied upstream (e.g. --filter-stage N2,N3 means
# only NREM windows exist at all, and NREM occurs in recurring bouts across
# a night -- so ANY subset of the valid window pool, including a random one,
# will show some apparent clustering just from that architecture). The only
# way to tell "genuinely more clustered than expected" apart from "just how
# N2/N3 naturally distributes across this recording" is to compare against
# a null model built from the SAME subject's own valid-window population.

def compute_temporal_clustering_statistic(times_sec: np.ndarray) -> float:
    """
    Summarizes how temporally clustered a set of window timestamps is via
    the median gap between consecutive timestamps (sorted ascending).
    Smaller -> more clustered (most windows have a close neighbor in time);
    larger -> more spread out.
    """
    times_sec = np.sort(np.asarray(times_sec, dtype=np.float64))
    if len(times_sec) < 2:
        return float("nan")
    return float(np.median(np.diff(times_sec)))


def run_temporal_clustering_null_test(
    contributor_times_sec: np.ndarray,
    population_times_sec: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Tests whether `contributor_times_sec` are more temporally clustered than
    expected by chance, given the full population of valid window
    timestamps (e.g. every N2/N3 window for this subject) they were drawn
    from. Draws `n_bootstrap` random same-size samples (without
    replacement) from `population_times_sec`, computes the same clustering
    statistic for each, and reports where the observed statistic falls in
    that null distribution.

    `empirical_p_value` is the fraction of random draws at least as
    clustered as what was actually observed (i.e. P(null stat <= observed
    stat)) -- small (e.g. < 0.05) means the real contributors are more
    tightly clustered in time than a random same-size sample of this
    subject's own valid windows would be, which is NOT explained away by
    stage-filtering/sleep-cycle architecture alone (that's already baked
    into the null model via `population_times_sec`).
    """
    rng = np.random.RandomState(seed)
    observed_stat = compute_temporal_clustering_statistic(contributor_times_sec)

    n = len(contributor_times_sec)
    population_times_sec = np.asarray(population_times_sec, dtype=np.float64)
    if n >= len(population_times_sec):
        print(
            "  [Warning] Contributor count >= population size -- the null model is degenerate "
            "(every random draw is just the full population), so this test is uninformative here."
        )

    null_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(population_times_sec, size=n, replace=False)
        null_stats[b] = compute_temporal_clustering_statistic(sample)

    return {
        "observed_median_gap_sec": observed_stat,
        "null_median_gap_mean_sec": float(np.mean(null_stats)),
        "null_median_gap_std_sec": float(np.std(null_stats)),
        "empirical_p_value": float(np.mean(null_stats <= observed_stat)),
        "n_bootstrap": n_bootstrap,
        "population_size": int(len(population_times_sec)),
        "sample_size": int(n)
    }


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
    contrib_group.add_argument(
        "--clustering-n-bootstrap", type=int, default=1000,
        help="Number of random same-size draws from each subject's full valid-window population used "
             "to build the null distribution for the temporal clustering test. Set to 0 to skip the test."
    )
    add_log_filename_argument(parser, __file__)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    seed_everything(args.seed)
    logger = setup_logger(args.log_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./pooling_contribution_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    window_sec = float(args.num_patches)

    # 1. Instantiate Model Architecture + load its checkpoint, metadata-first (same helper as
    # p09c/p09/p10), with checkpoint_kind deciding head-only vs. full-model loading deterministically.
    if args.features_pt:
        model, ckpt = build_frozen_probe(args, device, logger)
    else:
        model, ckpt = build_e2e_classifier(args, device, logger)
    ckpt_thresholds, _epoch, ckpt_pooling_params = extract_ckpt_metadata(ckpt)

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

    # SubjectEEGInspector's threshold only affects report['prediction'] (context for the export), not the
    # contribution computation itself (which depends only on pooling_strategy/top_percentile/t_window) --
    # but it still needs to be the REAL calibrated operating threshold, not an arbitrary placeholder, or
    # the exported "prediction" field will disagree with p09/p09c/p09d's (same bug class as
    # get_operating_threshold was written to prevent elsewhere in this pipeline).
    threshold = get_operating_threshold(
        pooling_strategy=pooling_strategy,
        override_threshold=args.override_threshold,
        ckpt_thresholds=ckpt_thresholds
    )
    inspector = SubjectEEGInspector(model=model, device=device, threshold=threshold)

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

        clustering_test = None
        if args.clustering_n_bootstrap > 0 and len(contributor_records) >= 2:
            contributor_times_sec = np.array([r["start_time_sec"] for r in contributor_records])
            population_times_sec = np.array(report["indices"], dtype=np.float64) * window_sec
            clustering_test = run_temporal_clustering_null_test(
                contributor_times_sec, population_times_sec,
                n_bootstrap=args.clustering_n_bootstrap, seed=args.seed
            )
            print(
                f"  Temporal clustering test: observed median gap={clustering_test['observed_median_gap_sec']:.1f}s "
                f"vs. null mean={clustering_test['null_median_gap_mean_sec']:.1f}s "
                f"(+/-{clustering_test['null_median_gap_std_sec']:.1f}s) "
                f"-> p={clustering_test['empirical_p_value']:.4f} "
                f"[population={clustering_test['population_size']} windows, "
                f"n_bootstrap={clustering_test['n_bootstrap']}]"
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
            "priority_tiers": {tier_name: contributor_records},
            "temporal_clustering_test": clustering_test
        }
        with open(json_path, "w") as f:
            json.dump(export_payload, f, indent=4)
        print(f"-> Saved pooling contributor windows: {json_path}")


if __name__ == "__main__":
    main()
