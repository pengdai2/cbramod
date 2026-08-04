#!/usr/bin/env python3
"""
Production CLI Pipeline for CBraMod with Manifest-based .npy Datasets.

Usage:
  # 1. First run (Parses manifest files, extracts backbone embeddings, and trains head):
  python cbramod_manifest_pipeline.py \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --data-dir /data/eeg_study/npy_files \
      --num-workers 8

  # 2. Subsequent runs (Auto-detects cached embeddings, skips .npy reading, trains head):
  python cbramod_manifest_pipeline.py --head-lr 0.0003 --epochs 40

  # 3. Force re-extraction from .npy files:
  python cbramod_manifest_pipeline.py \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --force-extract
"""

import argparse
import csv
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from einops.layers.torch import Rearrange
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

try:
    from braindecode.models import CBraMod
except ImportError:
    raise ImportError("The 'braindecode' library is required. Install via: pip install braindecode")

from real_world_benchmark import RealSleepEEGDataset


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
    data_group.add_argument("--train-manifest", type=str, default=None, help="Path to training manifest file (CSV/TSV/JSON/JSONL)")
    data_group.add_argument("--val-manifest", type=str, default=None, help="Path to validation manifest file (CSV/TSV/JSON/JSONL)")
    data_group.add_argument("--data-dir", type=str, default=None, help="Root directory containing .npy files (if paths in manifest are relative)")
    data_group.add_argument("--cache-dir", type=str, default="/opt/cbra_data/checkpoints", help="Directory for cached embeddings & checkpoints")
    data_group.add_argument("--force-extract", action="store_true", help="Force re-extraction of backbone embeddings even if cache exists")

    # Hyperparameters
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument("--epochs", type=int, default=40, help="Maximum training epochs for linear probe head")
    hp_group.add_argument("--batch-size", type=int, default=512, help="Batch size for head training and feature extraction")
    hp_group.add_argument("--head-lr", type=float, default=1e-4, help="Initial learning rate for the classification head")
    hp_group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing scheduler")
    hp_group.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay regularizer")
    hp_group.add_argument("--hidden-dim", type=int, default=128, help="Bottleneck linear layer dimension")
    hp_group.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in head")

    # System Controls
    sys_group = parser.add_argument_group("System Controls")
    sys_group.add_argument("--num-workers", type=int, default=4, help="DataLoader CPU workers for parallel .npy disk reads")
    sys_group.add_argument("--seed", type=int, default=42, help="Random seed for deterministic execution")
    sys_group.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without F1 improvement)")
    sys_group.add_argument("--num-channels", type=int, default=64, help="Number of EEG input channels")

    args = parser.parse_args()

    return PipelineConfig(
        cache_dir=Path(args.cache_dir),
        train_manifest_path=Path(args.train_manifest) if args.train_manifest else None,
        val_manifest_path=Path(args.val_manifest) if args.val_manifest else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        force_extract=args.force_extract,
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
    )


# =====================================================================
# 2. LOGGING & UTILITIES
# =====================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    """Configures structured logging to stdout and file."""
    logger = logging.getLogger("CBraModPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(log_path)
    f_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(f_handler)

    return logger


def seed_everything(seed: int = 42) -> None:
    """Ensures end-to-end reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 3. ARCHITECTURES
# =====================================================================

class CBraModFeatureExtractor(nn.Module):
    """Backbone extractor that channel-pools [B, C, S, P] -> [B, S, P]."""
    def __init__(self, num_channels: int = 64, sfreq: float = 200.0):
        super().__init__()
        self.backbone = CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained",
            n_chans=num_channels,
            sfreq=sfreq,
            return_encoder_output=True
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats.mean(dim=1)


class LinearProbeHead(nn.Module):
    """2-Layer MLP Head with LayerNorm and Dropout."""
    def __init__(self, num_patches: int = 30, emb_dim: int = 200, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        in_features = num_patches * emb_dim

        self.head = nn.Sequential(
            Rearrange("b s p -> b (s p)"),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# =====================================================================
# 4. EXTRACTION ENGINE
# =====================================================================

class EmbeddingManager:
    """Manages feature extraction from manifest-defined .npy files."""
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

    def extract_and_cache(self, manifest_path: Path, output_cache_path: Path, split_name: str) -> None:
        """Reads .npy files via RealSleepEEGDataset, extracts features, and saves unified tensor."""
        self.logger.info(f"[{split_name.upper()}] Initializing manifest dataset from: {manifest_path}")
        dataset = RealSleepEEGDataset(manifest_csv=manifest_path, data_dir=self.config.data_dir)
        self.logger.info(f"[{split_name.upper()}] Successfully parsed {len(dataset):,} .npy references.")

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

        all_embeddings, all_labels = [], []
        start_time = time.time()

        with torch.no_grad():
            for batch_x, batch_y in tqdm(loader, desc=f"Extracting {split_name}", unit="batch"):
                batch_x = batch_x.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=(self.config.use_amp and self.device.type == "cuda")):
                    pooled_feats = extractor(batch_x)

                all_embeddings.append(pooled_feats.cpu().float())
                all_labels.append(batch_y.cpu())

        cached_feats = torch.cat(all_embeddings, dim=0)
        cached_labels = torch.cat(all_labels, dim=0)

        torch.save({"feats": cached_feats, "labels": cached_labels}, output_cache_path)

        del extractor, dataset, loader, all_embeddings, all_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        file_size_mb = output_cache_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"✓ [{split_name.upper()}] Extraction complete ({elapsed:.1f}s) | Samples: {len(cached_feats):,} | Cache Size: {file_size_mb:.2f} MB")


# =====================================================================
# 5. TRAINER ENGINE WITH SCHEDULING & THRESHOLD OPTIMIZATION
# =====================================================================

class ProbeTrainer:
    """Trains classification head with Cosine Annealing and Decision Threshold Optimization."""
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

    def _find_best_threshold(self, val_targets: np.ndarray, val_probs: np.ndarray) -> Tuple[float, float]:
        """Sweeps decision thresholds [0.10, 0.90] to find max Macro F1."""
        best_threshold = 0.5
        best_f1 = 0.0

        thresholds = np.linspace(0.10, 0.90, 81)
        for t in thresholds:
            t_preds = (val_probs[:, 1] >= t).astype(int)
            _, _, t_f1, _ = precision_recall_fscore_support(val_targets, t_preds, average="macro", zero_division=0)
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_threshold = t

        return best_threshold, best_f1

    def train(self, train_cache_path: Path, val_cache_path: Path) -> float:
        self.logger.info("Loading cached feature tensors into RAM...")
        train_data = torch.load(train_cache_path, map_location="cpu")
        val_data = torch.load(val_cache_path, map_location="cpu")

        train_ds = TensorDataset(train_data["feats"], train_data["labels"])
        val_ds = TensorDataset(val_data["feats"], val_data["labels"])

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Inverse Class Weighting
        labels_np = train_data["labels"].numpy()
        class_counts = np.bincount(labels_np, minlength=self.config.num_classes)
        class_weights = torch.tensor(
            [len(labels_np) / (self.config.num_classes * count) for count in class_counts],
            dtype=torch.float
        ).to(self.device)

        head = LinearProbeHead(
            num_patches=self.config.num_patches,
            emb_dim=self.config.emb_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout_prob
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config.head_lr, weight_decay=self.config.weight_decay)
        
        # Cosine Annealing Learning Rate Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr)

        best_val_f1 = 0.0
        best_threshold = 0.5
        patience_counter = 0
        best_model_path = self.config.cache_dir / self.config.best_head_filename

        self.logger.info(
            f"Starting Probe Training ({self.config.epochs} Epochs Max | Batch Size: {self.config.batch_size} | "
            f"Head LR: {self.config.head_lr} -> {self.config.min_lr})"
        )
        self.logger.info("=" * 110)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            current_lr = scheduler.get_last_lr()[0]

            # Train Loop
            head.train()
            train_loss, train_correct = 0.0, 0
            for x_b, y_b in train_loader:
                x_b = x_b.to(self.device, non_blocking=True).float()  # Safe float32 conversion
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

            # Evaluate Loop
            head.eval()
            val_loss = 0.0
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b = x_b.to(self.device, non_blocking=True).float()  # Safe float32 conversion
                    y_b = y_b.to(self.device, non_blocking=True)

                    out = head(x_b)
                    loss = criterion(out, y_b)
                    probs = torch.softmax(out, dim=1)

                    val_loss += loss.item() * len(y_b)
                    val_preds.append(out.argmax(dim=1).cpu().numpy())
                    val_targets.append(y_b.cpu().numpy())
                    val_probs.append(probs.cpu().numpy())

            val_loss /= len(val_ds)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_probs = np.concatenate(val_probs)

            # Metrics at default 0.5 decision threshold
            val_acc = accuracy_score(val_targets, val_preds)
            bal_acc = balanced_accuracy_score(val_targets, val_preds)
            _, _, std_macro_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
            roc_auc = roc_auc_score(val_targets, val_probs[:, 1]) if self.config.num_classes == 2 else 0.5

            # Decision Threshold Optimization
            opt_thresh, tuned_macro_f1 = self._find_best_threshold(val_targets, val_probs) if self.config.num_classes == 2 else (0.5, std_macro_f1)
            
            # Step Learning Rate Scheduler
            scheduler.step()
            elapsed = time.time() - t0

            log_str = (
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.2f}s) | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {roc_auc:.4f} | "
                f"F1@0.5: {std_macro_f1:.4f} -> Tuned F1@{opt_thresh:.2f}: {tuned_macro_f1:.4f}"
            )

            # Checkpointing based on Tuned Macro F1
            if tuned_macro_f1 > best_val_f1:
                best_val_f1 = tuned_macro_f1
                best_threshold = opt_thresh
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": head.state_dict(),
                        "best_macro_f1": best_val_f1,
                        "optimal_threshold": best_threshold,
                    },
                    best_model_path
                )
                log_str += f" --> [BEST MODEL SAVED]"
            else:
                patience_counter += 1
                log_str += f" | EarlyStop: {patience_counter}/{self.config.early_stopping_patience}"

            self.logger.info(log_str)

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.logger.info("=" * 110)
        self.logger.info(f"Training Complete. Best Validation Macro F1: {best_val_f1:.4f} (at Optimal Threshold: {best_threshold:.2f})")
        return best_val_f1


# =====================================================================
# 6. PIPELINE ORCHESTRATOR
# =====================================================================

def main():
    config = parse_cli_args()
    seed_everything(config.seed)
    logger = setup_logger(config.cache_dir / config.log_filename)

    train_cache_path = config.cache_dir / config.train_cache_name
    val_cache_path = config.cache_dir / config.val_cache_name

    # Check Cache Existence
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

    # Phase 2: Fast Head Training with Scheduling & Thresholding
    trainer = ProbeTrainer(config, logger)
    trainer.train(train_cache_path, val_cache_path)


if __name__ == "__main__":
    main()