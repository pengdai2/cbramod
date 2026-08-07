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
import tqdm


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

    def plot_subject_diagnostics(self, report: Dict, save_path: Optional[Path] = None):
        """
        Renders a 4-panel diagnostic plot incorporating sleep stage hypnograms.
        """
        probs = report["window_probs"]
        stages = report["stages"]
        epochs = np.arange(len(probs))

        # Requirement 3: Map sleep stages to numerical ranks for hypnogram plotting
        stage_map = {"W": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4, "UNKNOWN": 5}
        numeric_stages = [stage_map.get(s.upper(), 5) for s in stages]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
        fig.suptitle(
            f"Subject Deep-Dive: {report['subject_id']} "
            f"(GT: {report['ground_truth']} | {report['pooling_strategy'].upper()}: {report['pooled_score']:.3f} | Pred: {report['prediction']})",
            fontsize=14, fontweight="bold"
        )

        # Panel 1: Chronological Probability Trajectory
        axes[0].plot(epochs, probs, color="navy", alpha=0.75, linewidth=1, label="P(Patient | Epoch)")
        axes[0].axhline(y=self.threshold, color="red", linestyle="--", label=f"Threshold ({self.threshold})")
        axes[0].axhline(y=report["pooled_score"], color="orange", linestyle=":", label=f"Pooled Score ({report['pooled_score']:.3f})")
        axes[0].set_ylabel("Probability")
        axes[0].set_title("Chronological Window Probability Trajectory")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Sleep Stage Hypnogram Alignment
        axes[1].step(epochs, numeric_stages, where="mid", color="purple", linewidth=1.5)
        axes[1].set_yticks([0, 1, 2, 3, 4, 5])
        axes[1].set_yticklabels(["Wake", "REM", "N1", "N2", "N3", "Unknown"])  # Focus on N2/N3 target
        axes[1].invert_yaxis()  # Standard clinical hypnogram orientation
        axes[1].set_ylabel("Stage")
        axes[1].set_title("Aligned Sleep Stage Hypnogram (Metadata)")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Probability Density (KDE/Hist)
        axes[2].hist(probs, bins=30, color="teal", edgecolor="black", alpha=0.7, density=True)
        axes[2].axvline(x=self.threshold, color="red", linestyle="--", label=f"Threshold ({self.threshold})")
        axes[2].set_xlabel("Probability")
        axes[2].set_ylabel("Density")
        axes[2].set_title("Epoch Probability Density (Tail Weight Analysis)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Panel 4: Sorted Epoch Profile (Scree Plot)
        sorted_probs = np.sort(probs)[::-1]
        axes[3].plot(sorted_probs, color="darkgreen", linewidth=2, label="P(Patient | Epoch)")
        axes[3].axhline(y=self.threshold, color="red", linestyle="--", label=f"Threshold ({self.threshold})")
        axes[3].set_xlabel("Sorted Epoch Rank")
        axes[3].set_ylabel("Probability")
        axes[3].set_title("Sorted Epoch Energy Profile (Tail Decay)")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"-> Saved diagnostic plot: {save_path}")
        plt.close()


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

    debugger = SubjectEEGInspector(model=model, device=device, threshold=threshold)

    # 3. Process filtered subjects
    for idx in tqdm(range(len(dataset)), desc="Process Subjects (Raw EEG)"):
        x_tensor, y_tensor, subj_id, stages = dataset[idx]
        if x_tensor.shape[0] == 0:
            print(f"Skipping {subj_id}: No valid windows after stage filtering.")
            continue

        report = debugger.inspect_subject(
            x_tensor=x_tensor,
            subject_id=subj_id,
            ground_truth=y_tensor.item(),
            stages=stages,
            pooling_strategy=args.pooling_strategy,
            batch_size=args.batch_size
        )

        save_path = output_dir / f"{subj_id}_diagnostic.png"
        debugger.plot_subject_diagnostics(report, save_path=save_path)


if __name__ == "__main__":
    main()