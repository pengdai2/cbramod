"""
p09f_morphology_score_correlation.py

Directly tests the question this whole investigation keeps circling back to:
is there ANY detectable relationship between morphological/spectral EEG
features and the model's window-level probability -- across many windows
and many subjects, not eyeballed one window at a time.

Unlike p10_feature_attribution.py (which explains one window's prediction
via Integrated Gradients) and p09e_pooling_contribution_analysis.py (which
identifies which windows drive the POOLED subject score), this script
doesn't try to explain the model at all -- it just correlates two
independently-measurable quantities for every valid window of every subject:
  1. The model's own window-level probability (requires the checkpoint).
  2. Relative band power in the classic bands, computed directly from the
     raw signal -- requires no model at all.

This script no longer reports YASA spindle/slow-wave event counts -- it
operates on the Z-scored channel-mean signal (see below), which YASA's
detectors are not calibrated for (real EEG amplitude, not Z-scored to
std~=1), making those counts unreliable independent of anything else.
p09k_absolute_band_power_analysis.py computes the same counts on the
reconstructed real-uV signal instead (needs metadata from a p02 re-slice
that persists per-channel norm_mean_uv/norm_std_uv) -- use that script's
YASA-count correlations, not a stale run of this one.

Feature computation operates on the CHANNEL-MEAN-POOLED signal (average
across all channels within a window), matching exactly what
CBraModE2EClassifier itself operates on after its own channel-mean pooling
step (`feats.mean(dim=1)`) -- so "does this feature predict the score" is
asked at the same level of spatial aggregation the model actually sees,
not a per-channel level the model's own averaging would dilute anyway.

Reports both:
  - Pooled correlation across all windows from all subjects together.
    Conflates within-subject and between-subject variance -- a subject
    with a generally elevated score AND elevated feature values throughout
    can drive this without it reflecting a real within-recording
    relationship.
  - Within-subject correlation, summarized across subjects. The more
    direct test of "why does THIS window score higher than THAT window,
    for the SAME subject" -- the actual motivating question behind this
    whole investigation.

Usage:
  python p09f_morphology_score_correlation.py \
      --probe-checkpoint model.pt --manifest test_manifest.csv \
      --output-dir results/ --max-windows-per-subject 200
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from tqdm import tqdm

from cbramod_common import (
    BAND_DEFS,
    PANSubjectEEGDataset,
    add_log_filename_argument,
    build_e2e_classifier,
    extract_ckpt_metadata,
    report_probability_correlations,
    resolve_pooling_config,
    seed_everything,
    setup_inference_cli_parser,
)
from cbramod_utils import setup_logger
from p09c_clinical_subject_diagnostics import SubjectEEGInspector, load_subject_ids_from_json


def compute_band_powers(signal_1d: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Welch relative band power for delta/theta/alpha/sigma/beta on a single
    channel-mean-pooled window. Relative (band power / total 0.5-30Hz power)
    rather than absolute, so it's comparable across subjects with different
    absolute amplitude scales (reference scheme, gain, etc.). Uses a raw
    power sum rather than trapezoidal integration -- the frequency-bin-width
    scaling factor is identical for every band since Welch returns evenly
    spaced bins, so it cancels exactly in the ratio.
    """
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=min(len(signal_1d), int(sfreq * 4)))
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = psd[total_mask].sum()
    powers = {}
    for band, (lo, hi) in BAND_DEFS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[f"{band}_relpower"] = float(psd[mask].sum() / total_power) if total_power > 0 else 0.0
    return powers


def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Morphology/Spectral Feature vs. Window Score Correlation")
    group = parser.add_argument_group("Morphology-Score Correlation")
    group.add_argument(
        "--max-windows-per-subject", type=int, default=None,
        help="Randomly subsample to at most this many windows per subject (for speed on long recordings). "
             "Default: use every valid window."
    )
    group.add_argument(
        "--subjects-json", type=str, default=None,
        help="Path to a p09d_subject_confidence_report.py --output-json report. Its subject_ids are "
             "unioned with --subject-id (if also given) to select which subjects to analyze."
    )
    group.add_argument(
        "--output-csv", type=str, default="morphology_score_correlation.csv",
        help="Filename (relative to --output-dir) for the full per-window feature table."
    )
    add_log_filename_argument(parser, __file__)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    logger = setup_logger(args.log_filename)
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./morphology_score_correlation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        raise ValueError(
            "This script needs raw EEG (--manifest) to compute band power features from the signal "
            "itself; --cache-dir (pre-extracted embeddings) has no raw waveform left to analyze."
        )

    model, ckpt = build_e2e_classifier(args, device, logger)
    _thresholds, _epoch, ckpt_pooling_params = extract_ckpt_metadata(ckpt)

    pooling_strategy, top_percentile, t_window = resolve_pooling_config(
        pooling_strategy=args.pooling_strategy, top_percentile=args.top_percentile,
        t_window=args.t_window, ckpt_pooling_params=ckpt_pooling_params
    )
    print(
        f"Pooling config: strategy={pooling_strategy}, top_percentile={top_percentile}, t_window={t_window} "
        f"(source: {'CLI override' if args.pooling_strategy is not None else 'checkpoint/default'})"
    )
    # Threshold doesn't matter here -- this script never uses report['prediction'], only the raw
    # window-level probabilities and the true label for context in the exported table.
    inspector = SubjectEEGInspector(model=model, device=device, threshold=0.5)

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

    all_rows: List[Dict] = []
    for idx in tqdm(range(len(dataset)), desc="Process Subjects"):
        x_tensor, y_tensor, subj_id, stages, indices = dataset[idx]
        if x_tensor.shape[0] == 0:
            print(f"Skipping {subj_id}: No valid windows after stage filtering.")
            continue

        report = inspector.inspect_subject(
            x_tensor=x_tensor, subject_id=subj_id, ground_truth=y_tensor.item(),
            stages=stages, indices=indices, pooling_strategy=pooling_strategy,
            top_percentile=top_percentile, t_window=t_window, batch_size=args.batch_size
        )

        num_windows = x_tensor.shape[0]
        window_order = np.arange(num_windows)
        if args.max_windows_per_subject and num_windows > args.max_windows_per_subject:
            window_order = rng.choice(num_windows, size=args.max_windows_per_subject, replace=False)

        raw_np = x_tensor.numpy()  # [N, C, T], same window order as report["window_probs"]/["stages"]/["indices"]
        for f_idx in window_order:
            channel_mean_signal = raw_np[f_idx].mean(axis=0)  # [T] -- matches the model's own channel pooling
            band_powers = compute_band_powers(channel_mean_signal, args.sfreq)

            all_rows.append({
                "subject_id": subj_id,
                "ground_truth": report["ground_truth"],
                "raw_epoch_index": int(report["indices"][f_idx]),
                "stage": report["stages"][f_idx] if f_idx < len(report["stages"]) else "UNKNOWN",
                "probability": float(report["window_probs"][f_idx]),
                **band_powers,
            })

    if not all_rows:
        print("No windows collected -- nothing to correlate.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} window-level rows across {df['subject_id'].nunique()} subjects to: {csv_path}")

    feature_cols = [c for c in df.columns if c not in
                     ("subject_id", "ground_truth", "raw_epoch_index", "stage", "probability")]

    report_probability_correlations(df, feature_cols, "ALL STAGES COMBINED")

    # Stage-stratified breakdown: a feature correlating with probability across a mix of stages
    # (e.g. N2 + N3) could just be reflecting a coarse stage effect -- delta power is definitionally
    # higher in N3, sigma/spindle power definitionally higher in N2, so "delta up / sigma down"
    # correlating with probability is exactly what you'd see if the model merely scores one stage
    # higher than the other, with no finer-grained relationship at all. Splitting by stage before
    # correlating checks whether the relationship survives *within* a single stage, which is the
    # more specific (and more interesting) claim.
    stages_present = sorted(s for s in df["stage"].unique() if s and s != "UNKNOWN")
    for stage in stages_present:
        report_probability_correlations(df[df["stage"] == stage], feature_cols, f"STAGE = {stage} ONLY")




if __name__ == "__main__":
    main()
