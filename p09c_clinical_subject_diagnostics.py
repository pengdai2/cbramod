import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

from cbramod_common import (
    CBraModE2EClassifier,
    CachedFeatureSubjectDataset,
    LinearProbeHead,
    MLPProbeHead,
    PANSubjectEEGDataset,
    compute_pooled_scores,
    get_operating_threshold,
    load_model_checkpoint,
    resolve_pooling_config,
    setup_inference_cli_parser
)
from cbramod_common import seed_everything


class SubjectEEGInspector:
    # Single source of truth for tier names + their plot style. Both
    # identify_priority_windows() (which produces these keys) and
    # plot_subject_eeg_diagnostics() (which looks styles up by key) iterate
    # this same mapping, so the two can't drift out of sync.
    TIER_STYLES: Dict[str, Dict] = {
        "Tier 1: Top Drivers": {"color": "#D62728", "marker": "*", "size": 130, "label": "Tier 1: Top Driver"},
        "Tier 2: Borderline":  {"color": "#9467BD", "marker": "D", "size": 70,  "label": "Tier 2: Borderline"},
        "Tier 3: Spikes":      {"color": "#FF7F0E", "marker": "^", "size": 90,  "label": "Tier 3: Temporal Spike"},
        "Tier 4: Baselines":   {"color": "#2CA02C", "marker": "s", "size": 70,  "label": "Tier 4: Baseline Control"},
    }

    def __init__(self, model: nn.Module, device: torch.device, threshold: float = 0.66):
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def inspect_subject(
        self,
        x_tensor: torch.Tensor,
        subject_id: str,
        ground_truth: int,
        stages: List[str],
        indices: List[int],
        pooling_strategy: str = "p85_score",
        top_percentile: float = 0.10,
        t_window: float = 0.60,
        batch_size: int = 64
    ) -> Dict:
        num_windows = x_tensor.shape[0]
        window_probs = []

        for j in range(0, num_windows, batch_size):
            x_batch = x_tensor[j : j + batch_size].to(self.device)
            logits = self.model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            window_probs.append(probs)

        probs_arr = np.concatenate(window_probs) if window_probs else np.array([])
        pooled_score = compute_pooled_scores(
            probs_arr, method=pooling_strategy, top_percentile=top_percentile, t_window=t_window
        )
        prediction = int(pooled_score >= self.threshold)

        return {
            "subject_id": subject_id,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "pooled_score": pooled_score,
            "is_correct": prediction == ground_truth,
            "total_windows": num_windows,
            "window_probs": probs_arr,
            "stages": stages,
            "indices": indices,
            "pooling_strategy": pooling_strategy
        }

    @staticmethod
    def _window_record(f_idx: int, probs: np.ndarray, raw_indices, stages: List[str], window_sec: float) -> Dict:
        """Builds the single per-window record shape shared by all tiers."""
        return {
            "filtered_index": int(f_idx),
            "raw_epoch_index": int(raw_indices[f_idx]),
            "probability": float(probs[f_idx]),
            "stage": stages[f_idx] if f_idx < len(stages) else "UNKNOWN",
            "start_time_sec": float(raw_indices[f_idx] * window_sec)
        }

    def _select_tier(
        self,
        candidate_order: np.ndarray,
        assigned_indices: set,
        top_k_per_tier: int,
        probs: np.ndarray,
        raw_indices,
        stages: List[str],
        window_sec: float,
        stop_predicate=None
    ) -> List[Dict]:
        """
        Walks `candidate_order` (indices already sorted by tier priority),
        skipping windows already claimed by an earlier tier, until either
        `top_k_per_tier` windows are collected or `stop_predicate` says stop.
        """
        selected: List[Dict] = []
        for f_idx in candidate_order:
            if f_idx in assigned_indices:
                continue
            if stop_predicate is not None and stop_predicate(f_idx):
                break
            if len(selected) >= top_k_per_tier:
                break
            selected.append(self._window_record(f_idx, probs, raw_indices, stages, window_sec))
            assigned_indices.add(f_idx)
        return selected

    def identify_priority_windows(
        self,
        report: Dict,
        top_k_per_tier: int = 3,
        window_sec: float = 30.0
    ) -> Tuple[Dict[str, List[Dict]], np.ndarray]:
        """
        Identifies target windows across 4 sampling tiers:
          - Tier 1: Top Driver Anchors (Highest probability windows)
          - Tier 2: Borderline Windows (Probability closest to operating threshold)
          - Tier 3: Temporal Spikes (Local probability peaks/outliers vs. neighbors)
          - Tier 4: Clean Baselines (Lowest probability windows)

        Returns a tuple of (tier_assignments, rank_order), where rank_order is
        the full probability-descending index order used for Tier 1 — callers
        that also need a probability ranking (e.g. the scree plot) should reuse
        it instead of re-sorting `window_probs` themselves.
        """
        probs = report["window_probs"]
        stages = report["stages"]
        raw_indices = report["indices"]
        n_windows = len(probs)

        if n_windows == 0:
            return {}, np.array([], dtype=int)

        assigned_indices = set()
        rank_order = np.argsort(probs)[::-1]  # Highest to lowest probability

        thresh_diffs = np.abs(probs - self.threshold)
        border_order = np.argsort(thresh_diffs)

        spike_scores = np.zeros(n_windows)
        for i in range(1, n_windows - 1):
            neighbor_avg = (probs[i - 1] + probs[i + 1]) / 2.0
            spike_scores[i] = probs[i] - neighbor_avg
        spike_order = np.argsort(spike_scores)[::-1]

        # Candidate order (+ optional stop predicate) per tier, positional against
        # TIER_STYLES so tier names live in exactly one place (TIER_STYLES).
        tier_candidates = [
            (rank_order, None),                                             # Tier 1: Top Drivers
            (border_order, None),                                           # Tier 2: Borderline
            (spike_order, lambda f_idx: spike_scores[f_idx] <= 0),          # Tier 3: Spikes (must be a positive local spike)
            (rank_order[::-1], None),                                       # Tier 4: Baselines (lowest to highest probability)
        ]

        tier_assignments = {
            tier_name: self._select_tier(
                candidate_order, assigned_indices, top_k_per_tier, probs, raw_indices, stages, window_sec,
                stop_predicate=stop_predicate
            )
            for tier_name, (candidate_order, stop_predicate) in zip(self.TIER_STYLES.keys(), tier_candidates)
        }

        return tier_assignments, rank_order

    def plot_subject_eeg_diagnostics(
        self,
        report: Dict,
        priority_windows: Optional[Dict[str, List[Dict]]] = None,
        rank_order: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Generates 4-Panel EEG Diagnostic Dashboard:
        - Panel 1: Hypnogram / Epoch Stage Timeline
        - Panel 2: Epoch Probability Sequence, with priority-tier scatter overlay
        - Panel 3: Probability Density (KDE/Hist), with priority-tier scatter overlay
        - Panel 4: Sorted Epoch Profile (Scree Plot), with priority-tier scatter overlay

        Args:
            report: Output of `SubjectEEGInspector.inspect_subject`.
            priority_windows: Tier-name -> list-of-window-record dict, as
                returned by `identify_priority_windows`. Tier names must be
                keys of `TIER_STYLES`. If omitted, no tier markers are drawn.
            rank_order: Probability-descending index order over
                `report["window_probs"]`, as returned by
                `identify_priority_windows`. Pass this through when available
                to avoid re-sorting `window_probs` for the Panel 4 scree plot;
                if omitted, it is computed here.
        """
        window_probs = report["window_probs"]
        stages = report["stages"]
        score = report["pooled_score"]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Subject Deep-Dive: {report['subject_id']} "
            f"(GT: {report['ground_truth']} | {report['pooling_strategy'].upper()}: {score:.3f} | Pred: {report['prediction']})",
            fontsize=14, fontweight="bold"
        )
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 3])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])

        n_epochs = len(window_probs)
        epoch_indices = np.arange(n_epochs)
        tier_style = self.TIER_STYLES

        # -------------------------------------------------------------------------
        # Panel 1: Hypnogram (Epoch Sleep Stages)
        # -------------------------------------------------------------------------
        # `stages` may be a numpy array (CachedFeatureSubjectDataset) or a
        # plain list (PANSubjectEEGDataset) -- `if stages` alone raises
        # "truth value of an array is ambiguous" for the former once it has
        # more than one element, so check length instead of truthiness.
        if len(stages) > 0 and len(stages) == n_epochs:
            stage_map = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0, "UNKNOWN": -1}
            num_stages = [stage_map.get(str(s).upper(), -1) for s in stages]
            ax1.step(epoch_indices, num_stages, where='mid', color='midnightblue', linewidth=1.5)
            ax1.set_yticks(list(stage_map.values()))
            ax1.set_yticklabels(list(stage_map.keys()))
            ax1.set_title("Panel 1: Sleep Stage Hypnogram")
            ax1.set_ylabel("Stage")
            ax1.set_xlabel("Epoch Index")
            ax1.grid(True, linestyle=':', alpha=0.6)
        else:
            ax1.text(0.5, 0.5, "Sleep Stage Metadata Unavailable", ha='center', va='center')
            ax1.set_title("Panel 1: Hypnogram")

        # -------------------------------------------------------------------------
        # Panel 2: Epoch Probability Sequence with Tier Annotations
        # -------------------------------------------------------------------------
        ax2.plot(epoch_indices, window_probs, color='slategray', alpha=0.6, linewidth=1.2, label='Epoch Prob')
        ax2.axhline(self.threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({self.threshold:.2f})')
        ax2.axhline(score, color='darkorange', linestyle='-', linewidth=1.5, label=f'Pooled Score ({score:.2f})')

        # Scatter overlay for priority tiers in Panel 2
        if priority_windows:
            for tier_name, windows in priority_windows.items():
                if not windows:
                    continue
                style = tier_style[tier_name]
                f_idxs = [w["filtered_index"] for w in windows]
                p_vals = [w["probability"] for w in windows]
                ax2.scatter(
                    f_idxs, p_vals,
                    color=style["color"], marker=style["marker"], s=style["size"],
                    zorder=6, edgecolor='black', linewidth=0.8, label=style["label"]
                )

        ax2.set_title("Panel 2: Window Probability Sequence & Priority Epochs")
        ax2.set_xlabel("Epoch Index")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right', ncol=2, fontsize='small')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # -------------------------------------------------------------------------
        # Panel 3: Probability Density (KDE/Hist) with Tier Overlays
        # -------------------------------------------------------------------------
        sns.histplot(window_probs, kde=True, ax=ax3, color='steelblue', bins=25, stat='density', alpha=0.4)
        ax3.axvline(self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold:.3f})')

        if report["pooling_strategy"] == "p85_score":
            ax3.axvline(score, color='darkorange', linestyle='-', linewidth=2, label=f'p85 Score ({score:.3f})')
            min_line, max_line = min(score, self.threshold), max(score, self.threshold)
            ax3.axvspan(min_line, max_line, color='gold', alpha=0.2, label='Margin Delta')

        if priority_windows:
            for tier_name, windows in priority_windows.items():
                if not windows:
                    continue
                style = tier_style[tier_name]
                p_vals = [w["probability"] for w in windows]
                ax3.scatter(
                    p_vals, np.zeros_like(p_vals),
                    color=style["color"], marker=style["marker"], s=style["size"],
                    zorder=6, edgecolor='black', clip_on=False
                )

        ax3.set_title("Panel 3: Probability Density (KDE/Hist)")
        ax3.set_xlabel("Window Probability")
        ax3.set_ylabel("Density")
        ax3.legend(loc='upper right', fontsize='small')
        ax3.grid(True, linestyle=':', alpha=0.6)

        # -------------------------------------------------------------------------
        # Panel 4: Sorted Epoch Profile (Scree Plot) with Tier Overlays
        # -------------------------------------------------------------------------
        # Reuse the rank order identify_priority_windows already computed (when
        # given) instead of re-sorting window_probs a second time.
        sort_perm = rank_order if rank_order is not None else np.argsort(window_probs)[::-1]  # High to Low sorting
        sorted_probs = window_probs[sort_perm]
        ranks = np.arange(1, n_epochs + 1)

        # Map filtered_index -> rank_index
        f_idx_to_rank = {f_idx: rank for rank, f_idx in zip(ranks, sort_perm)}

        ax4.plot(ranks, sorted_probs, color='teal', linewidth=2, label='Sorted Probs Profile')
        ax4.axhline(self.threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({self.threshold:.3f})')

        if report["pooling_strategy"] == "p85_score":
            ax4.axhline(score, color='darkorange', linestyle='-', linewidth=1.5, label=f'p85 Score ({score:.3f})')
            p85_rank_idx = int(np.ceil(n_epochs * (1.0 - 0.85)))
            p85_rank_idx = max(1, min(n_epochs, p85_rank_idx))
            ax4.scatter([p85_rank_idx], [score], color='darkorange', s=90, zorder=5, edgecolor='black', label=f'p85 Rank ({p85_rank_idx}/{n_epochs})')
            ax4.axvline(p85_rank_idx, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.7)

        # Scatter overlay on Scree Plot
        if priority_windows:
            for tier_name, windows in priority_windows.items():
                if not windows:
                    continue
                style = tier_style[tier_name]
                tier_ranks = [f_idx_to_rank[w["filtered_index"]] for w in windows]
                p_vals = [w["probability"] for w in windows]
                ax4.scatter(
                    tier_ranks, p_vals,
                    color=style["color"], marker=style["marker"], s=style["size"],
                    zorder=6, edgecolor='black', linewidth=0.8
                )

        ax4.set_title("Panel 4: Sorted Epoch Profile (Scree Plot)")
        ax4.set_xlabel("Epoch Rank (Sorted High to Low)")
        ax4.set_ylabel("Probability")
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(loc='upper right', fontsize='small')
        ax4.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"-> Saved diagnostic plot: {save_path}")
        plt.close()

        return fig


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

def load_subject_ids_from_json(json_path: Path) -> List[str]:
    """
    Extracts the unique subject_ids referenced anywhere in a
    p09d_subject_confidence_report.py --output-json report (misclassified +
    highest/lowest-confidence P/N selections), so that report can be handed
    straight to --subjects-json to run detailed per-window diagnostics on
    exactly the subjects it flagged.
    """
    with open(json_path) as f:
        payload = json.load(f)

    subject_ids: List[str] = [row["subject_id"] for row in payload.get("misclassified", [])]
    for bucket in ("highest_confidence", "lowest_confidence"):
        for class_rows in payload.get(bucket, {}).values():
            subject_ids.extend(row["subject_id"] for row in class_rows)

    # Dedupe while preserving first-seen order (a subject can legitimately
    # appear in both e.g. "highest_confidence" and a later --output-json call
    # over a different strategy's CSV, if the two were concatenated by hand).
    seen = set()
    unique_ids = []
    for sid in subject_ids:
        if sid not in seen:
            seen.add(sid)
            unique_ids.append(sid)
    return unique_ids


def parse_cli_args()-> argparse.Namespace:
    parser = setup_inference_cli_parser(description="Multi-Class Patient-Level Clinical Inference")
    parser.add_argument(
        "--subjects-json", type=str, default=None,
        help="Path to a p09d_subject_confidence_report.py --output-json report. Its "
             "misclassified/highest_confidence/lowest_confidence subject_ids are unioned with "
             "--subject-id (if also given) to select which subjects to run diagnostics on."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Model Architecture
    if args.features_pt:
        print("Instantiating isolated Probe Head for cached feature inference.")
        if args.head_type == "linear":
            model = LinearProbeHead(
                num_patches=args.num_patches,
                emb_dim=args.cbra_dim,
                num_classes=args.num_classes
            )
        else:
            model = MLPProbeHead(
                num_patches=args.num_patches,
                emb_dim=args.cbra_dim,
                hidden_dim=args.head_dim,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
    else:
        print("Instantiating full CBraModE2EClassifier for raw waveform inference.")
        model = CBraModE2EClassifier(
            num_channels=args.num_channels,
            sfreq=args.sfreq,
            num_patches=args.num_patches,
            emb_dim=args.cbra_dim,
            hidden_dim=args.head_dim,
            num_classes=args.num_classes,
            head_type=args.head_type
        )

    # 2. Load Model Checkpoint
    model, ckpt_thresholds, epoch, ckpt_pooling_params = load_model_checkpoint(model, Path(args.probe_checkpoint), device)
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
            manifest_csv=args.manifest,
            data_dir=args.data_dir,
            filter_stage=args.filter_stage,
            filter_subject=subject_filter,
            memory_map=True
        )
        print(f"Loaded raw EEG recording dataset for {len(dataset)} subjects.")

    # 4. Process subjects, extract 4-tier target windows, plot & export JSON
    for idx in tqdm(range(len(dataset)), desc="Process Subjects (Raw EEG)"):
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

        # 1) Extract all 4 tiers of target windows
        priority_windows, rank_order = inspector.identify_priority_windows(report, top_k_per_tier=3)

        # 2) Save diagnostic plot with distinctive tier markers
        save_path = output_dir / f"{subj_id}_diagnostic.png"
        inspector.plot_subject_eeg_diagnostics(
            report, priority_windows=priority_windows, rank_order=rank_order, save_path=save_path
        )

        # 3) Save priority windows to JSON for downstream attribution pipeline
        json_path = output_dir / f"{subj_id}_priority_windows.json"
        export_payload = {
            "subject_id": subj_id,
            "ground_truth": report["ground_truth"],
            "prediction": report["prediction"],
            "pooled_score": float(report["pooled_score"]),
            "threshold": float(threshold),
            "priority_tiers": priority_windows
        }
        with open(json_path, "w") as f:
            json.dump(export_payload, f, indent=4)
        print(f"-> Saved priority attribution windows: {json_path}")


if __name__ == "__main__":
    main()