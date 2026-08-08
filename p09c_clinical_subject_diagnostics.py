#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cbramod_utils import seed_everything
from cbramod_common import (
    CBraModE2EClassifier,
    PANSubjectEEGDataset,
    compute_pooled_scores,
    get_operating_threshold,
    setup_common_cli_parser,
    load_model_checkpoint
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import seaborn as sns


class SubjectEEGInspector:
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
        pooling_strategy: str = "p85_score",
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
        pooled_score = compute_pooled_scores(probs_arr, method=pooling_strategy)
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
            "pooling_strategy": pooling_strategy
        }

    def plot_subject_eeg_diagnostics(
        self,
        report: Dict,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Generates 4-Panel EEG Diagnostic Dashboard:
        - Panel 1: Hypnogram / Epoch Stage Timeline
        - Panel 2: Epoch Probability Sequence
        - Panel 3: Probability Density (KDE/Hist) highlighting p85 vs. Threshold
        - Panel 4: Sorted Epoch Profile (Scree Plot) highlighting p85 rank & threshold
        """
        window_probs = report["window_probs"]
        stages = report["stages"]
        score = report["pooled_score"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Subject Deep-Dive: {report["subject_id"]} "
            f"(GT: {report['ground_truth']} | {report['pooling_strategy'].upper()}: {score:.3f} | Pred: {report['prediction']})",
            fontsize=14, fontweight="bold"
        )
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        n_epochs = len(window_probs)
        epoch_indices = np.arange(n_epochs)

        # -------------------------------------------------------------------------
        # Panel 1: Hypnogram (Epoch Sleep Stages)
        # -------------------------------------------------------------------------
        if stages and len(stages) == n_epochs:
            stage_map = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0, "UNKNOWN": -1}
            num_stages = [stage_map.get(str(s).upper(), -1) for s in stages]
            ax1.step(epoch_indices, num_stages, where='mid', color='midnightblue', linewidth=1.5)
            ax1.set_yticks(list(stage_map.values()))
            ax1.set_yticklabels(list(stage_map.keys()))
            ax1.set_title(f"Panel 1: Sleep Stage Hypnogram")
            ax1.set_ylabel("Stage")
            ax1.set_xlabel("Epoch Index")
            ax1.grid(True, linestyle=':', alpha=0.6)
        else:
            ax1.text(0.5, 0.5, "Sleep Stage Metadata Unavailable", ha='center', va='center')
            ax1.set_title("Panel 1: Hypnogram")

        # -------------------------------------------------------------------------
        # Panel 2: Epoch Probability Sequence (Probability overlay across time)
        # -------------------------------------------------------------------------
        ax2.plot(epoch_indices, window_probs, color='slategray', alpha=0.7, label='Epoch Prob')
        ax2.axhline(self.threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({self.threshold:.2f})')
        ax2.axhline(score, color='darkorange', linestyle='-', linewidth=1.5, label=f'Pooled Score ({score:.2f})')
        ax2.set_title("Panel 2: Window Probability Sequence")
        ax2.set_xlabel("Epoch Index")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # -------------------------------------------------------------------------
        # Panel 3: Probability Density (KDE/Hist)
        # -------------------------------------------------------------------------
        sns.histplot(window_probs, kde=True, ax=ax3, color='steelblue', bins=25, stat='density', alpha=0.4)
        ax3.axvline(self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold:.3f})')

        if report["pooling_strategy"] == "p85_score":
            ax3.axvline(score, color='darkorange', linestyle='-', linewidth=2, label=f'p85 Score ({score:.3f})')
        
            # Shade region between threshold and p85 to emphasize divergence
            min_line, max_line = min(score, self.threshold), max(score, self.threshold)
            ax3.axvspan(min_line, max_line, color='gold', alpha=0.2, label='Margin Delta')
        
        ax3.set_title("Panel 3: Probability Density (KDE/Hist)")
        ax3.set_xlabel("Window Probability")
        ax3.set_ylabel("Density")
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle=':', alpha=0.6)

        # -------------------------------------------------------------------------
        # Panel 4: Sorted Epoch Profile (Scree Plot) - p85 Rank & Line Highlight
        # -------------------------------------------------------------------------
        sorted_probs = np.sort(window_probs)[::-1]  # Sort probabilities descending
        ranks = np.arange(1, n_epochs + 1)

        ax4.plot(ranks, sorted_probs, color='teal', linewidth=2, label='Sorted Probs Profile')
        ax4.axhline(self.threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({self.threshold:.3f})')

        if report["pooling_strategy"] == "p85_score":
            ax4.axhline(score, color='darkorange', linestyle='-', linewidth=1.5, label=f'p85 Score ({score:.3f})')

            # 85th percentile rank position (15th percentile from the top of sorted values)
            p85_rank_idx = int(np.ceil(n_epochs * (1.0 - 0.85)))
            p85_rank_idx = max(1, min(n_epochs, p85_rank_idx))

            # Highlight point where p85 intersects sorted profile
            ax4.scatter([p85_rank_idx], [score], color='darkorange', s=90, zorder=5, edgecolor='black',label=f'p85 Rank ({p85_rank_idx}/{n_epochs})')
        
            # Vertical guideline pointing to p85 rank index
            ax4.axvline(p85_rank_idx, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.7)

        ax4.set_title("Panel 4: Sorted Epoch Profile (Scree Plot)")
        ax4.set_xlabel("Epoch Rank (Sorted High to Low)")
        ax4.set_ylabel("Probability")
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(loc='upper right')
        ax4.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"-> Saved diagnostic plot: {save_path}")
        plt.close()

        return fig


# -----------------------------------------------------------------------------
# 4. CLI Argument Parser (Requirement 2)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run subject-level epoch diagnostics on specific clinical EEG cohorts."
    )
    setup_common_cli_parser(parser)

    subject_group = parser.add_argument_group("Subject Selection")
    subject_group.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV file.")
    subject_group.add_argument("--subject-id", type=str, default=None, help="Optional comma-separated list of specific Subject IDs to analyze (e.g., GRINS0322,GRINS0038).")

    ckpt_group = parser.add_argument_group("Model Checkpoint")
    ckpt_group.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # Pooling Strategy
    pool_group = parser.add_argument_group("Pooling Strategy")
    pool_group.add_argument(
        "--pooling-strategy", 
        type=str, 
        default="p85_score", 
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio", "all"],
        help="Pooling strategy choice (default: 'p85_score', or 'all' for full comparative report)"
    )
    pool_group.add_argument("--top-percentile", type=float, default=0.10, help="Top percentile ratio (default: 0.10)")
    pool_group.add_argument("--t-window", type=float, default=0.60, help="Window threshold for burden ratio (default: 0.60)")

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--override-threshold", type=float, default=None, help="Override operating decision threshold")
    misc_group.add_argument("--batch-size", type=int, default=512, help="Batch size for inference (default: 512)")
    misc_group.add_argument("--output-dir", type=str, default=None, help="Output directory for the subject analysis")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 5. Main Execution Loop
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse optional subject ID list
    subject_ids = [s.strip() for s in args.subject_id.split(",")] if args.subject_id else None

    # 1. Instantiate Dataset with subject filter
    dataset = PANSubjectEEGDataset(
        manifest_csv=args.manifest,
        data_dir=args.data_dir,
        filter_subject=subject_ids,
        filter_stage=args.filter_stage
    )

    print(f"Loaded {len(dataset)} matching subject(s) for diagnostic inspection.")

    # 2. Model Wrapper / Checkpoint Loader
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

    # 2. Load Model Checkpoint (Head-Only or Full-Model)
    model, ckpt_thresholds, epoch = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    threshold = get_operating_threshold(
        pooling_strategy=args.pooling_strategy,
        override_threshold=args.override_threshold,
        ckpt_thresholds=ckpt_thresholds
    )

    inspector = SubjectEEGInspector(model=model, device=device, threshold=threshold)

    # 3. Process filtered subjects
    for idx in tqdm(range(len(dataset)), desc="Process Subjects (Raw EEG)"):
        x_tensor, y_tensor, subj_id, stages = dataset[idx]
        if x_tensor.shape[0] == 0:
            print(f"Skipping {subj_id}: No valid windows after stage filtering.")
            continue

        report = inspector.inspect_subject(
            x_tensor=x_tensor,
            subject_id=subj_id,
            ground_truth=y_tensor.item(),
            stages=stages,
            pooling_strategy=args.pooling_strategy,
            batch_size=args.batch_size
        )

        save_path = output_dir / f"{subj_id}_diagnostic.png"
        inspector.plot_subject_eeg_diagnostics(report, save_path=save_path)


if __name__ == "__main__":
    main()