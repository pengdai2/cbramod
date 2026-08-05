"""
Production CLI Pipeline for CBraMod with Manifest-based .npy Datasets.
Supports Stage-Filtered Embedding Extraction, Window-Level Probe Training & 
Subject-Level Multi-Strategy Pooling Evaluation.

Usage:
  # 1. First run (Extracts embeddings filtered by stage & performs probe training):
  python cbramod_manifest_pipeline.py \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --data-dir /data/eeg_study/npy_files \
      --filter-stage N2,N3 \
      --num-workers 8

  # 2. Subsequent runs (Uses cached embeddings, trains head with custom primary pooling):
  python cbramod_manifest_pipeline.py --head-lr 0.0003 --epochs 40 --primary-pooling p85_score

  # 3. Force re-extraction from .npy files:
  python cbramod_manifest_pipeline.py \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --filter-stage N2,N3 \
      --force-extract
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from cbramod_utils import seed_everything, setup_logger
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from cbramod_common import CBraModFeatureExtractor, LinearProbeHead, PANSleepEEGDataset, compute_pooled_scores, setup_data_loader_and_criterion


# =====================================================================
# 1. CONFIGURATION & CLI PARSER
# =====================================================================

@dataclass
class PipelineConfig:
    """Dataclass storing pipeline execution state."""
    cache_dir: Path
    train_manifest_path: Optional[Path] = None
    val_manifest_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    train_cache_name: str = "cached_train_embeddings.pt"
    val_cache_name: str = "cached_val_embeddings.pt"
    best_head_filename: str = "cbramod_head_best.pt"
    log_filename: str = "pipeline.log"
    
    # Model & Feature Dimensions
    num_channels: int = 64
    sfreq: float = 200.0
    num_patches: int = 30
    emb_dim: int = 200
    num_classes: int = 2

    imbalance_strategy: str = "loss_weights"
    filter_stage: Optional[str] = None
    
    # Pooling & Threshold Hyperparameters
    primary_pooling: str = "p85_score"
    top_percentile: float = 0.10
    t_window: float = 0.60
    
    # Hyperparameters
    batch_size: int = 512
    epochs: int = 40
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    hidden_dim: int = 128
    dropout_prob: float = 0.3
    
    # Control Flags
    force_extract: bool = False
    num_workers: int = 4
    seed: int = 42
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.train_manifest_path:
            self.train_manifest_path = Path(self.train_manifest_path)
        if self.val_manifest_path:
            self.val_manifest_path = Path(self.val_manifest_path)
        if self.data_dir:
            self.data_dir = Path(self.data_dir)


def parse_cli_args() -> PipelineConfig:
    """Parses command-line arguments into a structured PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="CBraMod Automated Embedding Extraction & Probe Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Manifest & Data Paths
    data_group = parser.add_argument_group("Data Sources")
    data_group.add_argument("--train-manifest", type=str, default=None, help="Path to training manifest file (CSV/TSV/JSON)")
    data_group.add_argument("--val-manifest", type=str, default=None, help="Path to validation manifest file (CSV/TSV/JSON)")
    data_group.add_argument("--data-dir", type=str, default=None, help="Root directory containing .npy files")
    data_group.add_argument("--cache-dir", type=str, default="/opt/cbra_data/checkpoints", help="Directory for cached embeddings & checkpoints")
    data_group.add_argument("--force-extract", action="store_true", help="Force re-extraction of backbone embeddings")

    # Imbalance & Stage Controls
    strat_group = parser.add_argument_group("Imbalance & Stage Controls")
    strat_group.add_argument(
        "--imbalance-strategy", 
        type=str, 
        choices=["sampler", "loss_weights", "none"], 
        default="loss_weights", 
        help="Class imbalance handling: 'sampler' (WeightedRandomSampler), 'loss_weights' (Class-Weighted CrossEntropy), or 'none'"
    )
    strat_group.add_argument("--filter-stage", type=str, default="N2,N3", help="Comma-separated sleep stages to pass into PANSleepEEGDataset (e.g., N2,N3)")

    # Pooling Configurations
    pool_group = parser.add_argument_group("Subject-Level Pooling Options")
    pool_group.add_argument(
        "--primary-pooling", 
        type=str, 
        default="p85_score", 
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"],
        help="Primary pooling strategy used for early stopping and model selection"
    )
    pool_group.add_argument("--top-percentile", type=float, default=0.10, help="Top percentile ratio for top-K pooling methods")
    pool_group.add_argument("--t-window", type=float, default=0.60, help="Window threshold for pathology burden ratio")

    # Hyperparameters
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument("--epochs", type=int, default=40, help="Maximum training epochs for linear probe head")
    hp_group.add_argument("--batch-size", type=int, default=512, help="Batch size for training and feature extraction")
    hp_group.add_argument("--head-lr", type=float, default=1e-4, help="Initial learning rate for classification head")
    hp_group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing scheduler")
    hp_group.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay regularizer")
    hp_group.add_argument("--hidden-dim", type=int, default=128, help="Bottleneck linear layer dimension")
    hp_group.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in head")
    hp_group.add_argument("--num-classes", type=int, default=2, help="Number of target classes")

    # System Controls
    sys_group = parser.add_argument_group("System Controls")
    sys_group.add_argument("--num-workers", type=int, default=4, help="DataLoader CPU workers for disk reads")
    sys_group.add_argument("--seed", type=int, default=42, help="Random seed for deterministic execution")
    sys_group.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without Subject F1 improvement)")
    sys_group.add_argument("--num-channels", type=int, default=64, help="Number of EEG input channels")

    args = parser.parse_args()

    return PipelineConfig(
        cache_dir=Path(args.cache_dir),
        train_manifest_path=Path(args.train_manifest) if args.train_manifest else None,
        val_manifest_path=Path(args.val_manifest) if args.val_manifest else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        filter_stage=args.filter_stage,
        force_extract=args.force_extract,
        primary_pooling=args.primary_pooling,
        top_percentile=args.top_percentile,
        t_window=args.t_window,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout,
        seed=args.seed,
        early_stopping_patience=args.patience,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
    )


# =====================================================================
# 2. EMBEDDING EXTRACTION ENGINE WITH STAGE FILTERING & SUBJECT ID TRACKING
# =====================================================================

class EmbeddingManager:
    """Manages feature extraction from manifest .npy files with optional stage filtering."""
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

    def extract_and_cache(self, manifest_path: Path, output_cache_path: Path, split_name: str) -> None:
        """Reads .npy files, applies stage filtering, extracts backbone embeddings, and caches feats + labels + subject_ids."""
        filter_str = f" [Filter: {self.config.filter_stage}]" if self.config.filter_stage else ""
        self.logger.info(f"[{split_name.upper()}] Initializing PANSleepEEGDataset from: {manifest_path}{filter_str}")
        
        dataset = PANSleepEEGDataset(
            manifest_csv=manifest_path, 
            data_dir=self.config.data_dir,
            filter_stage=self.config.filter_stage
        )
        self.logger.info(f"[{split_name.upper()}] Successfully indexed {len(dataset):,} valid stage-filtered window references.")

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )

        self.logger.info(f"[{split_name.upper()}] Extracting backbone representations to: {output_cache_path}")
        extractor = CBraModFeatureExtractor(
            num_channels=self.config.num_channels,
            sfreq=self.config.sfreq
        ).to(self.device)
        extractor.eval()

        all_embeddings, all_labels, all_subject_ids = [], [], []
        start_time = time.time()

        with torch.no_grad():
            for batch_x, batch_y, batch_subj in tqdm(loader, desc=f"Extracting {split_name}", unit="batch"):
                batch_x = batch_x.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=(self.config.use_amp and self.device.type == "cuda")):
                    pooled_feats = extractor(batch_x)

                all_embeddings.append(pooled_feats.cpu().float())
                all_labels.append(batch_y.cpu())
                all_subject_ids.extend(batch_subj)

        cached_feats = torch.cat(all_embeddings, dim=0)
        cached_labels = torch.cat(all_labels, dim=0)

        torch.save({
            "feats": cached_feats, 
            "labels": cached_labels,
            "subject_ids": all_subject_ids
        }, output_cache_path)

        del extractor, dataset, loader, all_embeddings, all_labels, all_subject_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        file_size_mb = output_cache_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"✓ [{split_name.upper()}] Extraction complete ({elapsed:.1f}s) | Windows: {len(cached_feats):,} | Cache Size: {file_size_mb:.2f} MB")


# =====================================================================
# 3. TRAINER ENGINE WITH WINDOW TRAINING & FAST SUBJECT POOLING VAL
# =====================================================================

class ProbeTrainer:
    """Trains head on window instances while evaluating and tuning thresholds on subject-level pooled scores."""
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

    def _evaluate_subject_pooling(
        self, 
        val_probs: np.ndarray, 
        val_targets: np.ndarray, 
        val_subject_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Groups window probabilities by subject using an O(N) pre-indexed map, 
        applies all 4 pooling strategies, and performs threshold tuning.
        """
        strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]

        # O(N) linear indexing pass to pre-group window indices by subject ID
        subj_to_indices = defaultdict(list)
        for idx, subj in enumerate(val_subject_ids):
            subj_to_indices[subj].append(idx)

        subject_data = {strat: [] for strat in strategies}
        subject_labels = []

        # Iterate through pre-grouped subject slices
        for subj, idxs in subj_to_indices.items():
            idx_arr = np.array(idxs, dtype=np.int64)
            subj_probs = val_probs[idx_arr]
            subj_gt = val_targets[idx_arr[0]]
            subject_labels.append(subj_gt)

            for strat in strategies:
                if self.config.num_classes == 2:
                    score = compute_pooled_scores(
                        subj_probs[:, 1], 
                        method=strat, 
                        top_percentile=self.config.top_percentile, 
                        t_window=self.config.t_window
                    )
                else:
                    score = compute_pooled_scores(
                        subj_probs, 
                        method=strat, 
                        top_percentile=self.config.top_percentile, 
                        t_window=self.config.t_window
                    )
                subject_data[strat].append(score)

        subject_labels = np.array(subject_labels)
        results = {}

        # Strategy evaluation and threshold optimization
        for strat in strategies:
            scores = np.array(subject_data[strat])
            
            if self.config.num_classes == 2:
                # Binary threshold sweep to maximize Macro F1 on Subject predictions
                best_t = 0.5
                best_f1 = 0.0
                thresholds = np.linspace(0.01, 0.99, 99)
                
                for t in thresholds:
                    preds = (scores >= t).astype(int)
                    f1 = f1_score(subject_labels, preds, average="macro", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
                
                final_preds = (scores >= best_t).astype(int)
                acc = accuracy_score(subject_labels, final_preds)
                roc_auc = roc_auc_score(subject_labels, scores) if len(np.unique(subject_labels)) > 1 else 0.5

                results[strat] = {
                    "subject_macro_f1": best_f1,
                    "optimal_threshold": float(best_t),
                    "subject_accuracy": acc,
                    "roc_auc": roc_auc
                }
            else:
                # Multi-class argmax selection
                preds = np.argmax(scores, axis=1)
                macro_f1 = f1_score(subject_labels, preds, average="macro", zero_division=0)
                acc = accuracy_score(subject_labels, preds)
                results[strat] = {
                    "subject_macro_f1": macro_f1,
                    "optimal_threshold": 0.5,
                    "subject_accuracy": acc,
                    "roc_auc": 0.5
                }

        return results

    def train(self, train_cache_path: Path, val_cache_path: Path) -> float:
        self.logger.info("Loading cached feature tensors into RAM...")
        train_data = torch.load(train_cache_path, map_location="cpu")
        val_data = torch.load(val_cache_path, map_location="cpu")

        train_ds = TensorDataset(train_data["feats"], train_data["labels"])
        val_ds = TensorDataset(val_data["feats"], val_data["labels"])

        train_loader, criterion = setup_data_loader_and_criterion(
            dataset=train_ds,
            labels=train_data["labels"].numpy(),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            imbalance_strategy=self.config.imbalance_strategy,
            device=self.device,
            logger=self.logger
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

        val_subject_ids = val_data.get("subject_ids", [str(i) for i in range(len(val_ds))])

        head = LinearProbeHead(
            num_patches=self.config.num_patches,
            emb_dim=self.config.emb_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout_prob
        ).to(self.device)

        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config.head_lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr)

        best_primary_f1 = 0.0
        best_thresholds = {}
        patience_counter = 0
        best_model_path = self.config.cache_dir / self.config.best_head_filename

        self.logger.info(
            f"Starting Probe Training ({self.config.epochs} Epochs Max | Batch Size: {self.config.batch_size} | "
            f"Primary Pooling: {self.config.primary_pooling})"
        )
        self.logger.info("=" * 125)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            current_lr = scheduler.get_last_lr()[0]

            # 1. Window-Level Training Loop
            head.train()
            train_loss, train_correct = 0.0, 0
            for x_b, y_b in train_loader:
                x_b = x_b.to(self.device, non_blocking=True).float()
                y_b = y_b.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                out = head(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(y_b)
                train_correct += (out.argmax(dim=1) == y_b).sum().item()

            train_acc = train_correct / len(train_ds)
            train_loss /= len(train_ds)

            # 2. Window-Level Inference on Validation
            head.eval()
            val_loss = 0.0
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b = x_b.to(self.device, non_blocking=True).float()
                    y_b = y_b.to(self.device, non_blocking=True)

                    out = head(x_b)
                    loss = criterion(out, y_b)
                    probs = torch.softmax(out, dim=1)

                    val_loss += loss.item() * len(y_b)
                    val_preds.append(out.argmax(dim=1).cpu().numpy())
                    val_targets.append(y_b.cpu().numpy())
                    val_probs.append(probs.cpu().numpy())

            val_loss /= len(val_ds)
            val_targets = np.concatenate(val_targets)
            val_probs = np.concatenate(val_probs)

            # 3. Fast Subject-Level Multi-Strategy Pooling & Threshold Calibration
            pooling_results = self._evaluate_subject_pooling(val_probs, val_targets, val_subject_ids)
            
            primary_metrics = pooling_results[self.config.primary_pooling]
            primary_f1 = primary_metrics["subject_macro_f1"]
            primary_t = primary_metrics["optimal_threshold"]
            primary_acc = primary_metrics["subject_accuracy"]

            scheduler.step()
            elapsed = time.time() - t0

            # Format log string showing window loss + subject-level pooled metrics across strategies
            log_str = (
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.2f}s) | LR: {current_lr:.2e} | "
                f"Win Loss: {train_loss:.4f} | Subj Acc: {primary_acc*100:.2f}% | "
                f"Subj F1 ({self.config.primary_pooling}@{primary_t:.2f}): {primary_f1:.4f}"
            )

            # Model Selection & Checkpointing based on Primary Subject-Level Macro F1
            if primary_f1 > best_primary_f1:
                best_primary_f1 = primary_f1
                patience_counter = 0
                best_thresholds = {strat: res["optimal_threshold"] for strat, res in pooling_results.items()}
                
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": head.state_dict(),
                        "best_macro_f1": best_primary_f1,
                        "primary_pooling": self.config.primary_pooling,
                        "optimal_thresholds": best_thresholds,
                        "pooling_summary": pooling_results
                    },
                    best_model_path
                )
                log_str += f" --> [BEST SUBJECT MODEL SAVED]"
            else:
                patience_counter += 1
                log_str += f" | EarlyStop: {patience_counter}/{self.config.early_stopping_patience}"

            self.logger.info(log_str)

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.logger.info("=" * 125)
        self.logger.info(
            f"Training Complete. Best Validation Subject Macro F1 ({self.config.primary_pooling}): {best_primary_f1:.4f}"
        )
        self.logger.info(f"Calibrated Strategy Thresholds: {best_thresholds}")
        return best_primary_f1


# =====================================================================
# 4. PIPELINE ORCHESTRATOR
# =====================================================================

def main():
    config = parse_cli_args()
    seed_everything(config.seed)
    logger = setup_logger(config.cache_dir / config.log_filename)

    train_cache_path = config.cache_dir / config.train_cache_name
    val_cache_path = config.cache_dir / config.val_cache_name

    cache_exists = train_cache_path.exists() and val_cache_path.exists()

    if not cache_exists or config.force_extract:
        logger.info("Cached embeddings not found or --force-extract specified. Initializing Feature Extraction Phase...")

        if not config.train_manifest_path or not config.val_manifest_path:
            logger.error("Missing manifest paths! Please provide --train-manifest and --val-manifest to extract features.")
            sys.exit(1)

        extractor_mgr = EmbeddingManager(config, logger)
        extractor_mgr.extract_and_cache(config.train_manifest_path, train_cache_path, split_name="train")
        extractor_mgr.extract_and_cache(config.val_manifest_path, val_cache_path, split_name="val")
    else:
        logger.info(f"Found existing cached feature files at '{config.cache_dir}'. Skipping .npy extraction phase.")

    # Phase 2: Probe Training on Windows + Multi-Strategy Subject Pooling Validation
    trainer = ProbeTrainer(config, logger)
    trainer.train(train_cache_path, val_cache_path)


if __name__ == "__main__":
    main()
