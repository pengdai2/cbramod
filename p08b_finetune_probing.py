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
import gc
import logging
from pathlib import Path
import sys
import time

import numpy as np
from cbramod_utils import seed_everything, setup_logger
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from cbramod_common import (
    CBraModFeatureExtractor,
    CBraModTrainer,
    LinearProbeHead,
    MLPProbeHead,
    PANSleepEEGDataset,
    setup_data_loader_and_criterion,
    setup_training_cli_parser
)


# =====================================================================
# 1. CONFIGURATION & CLI PARSER
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for the CBraMod manifest-based pipeline."""
    parser = setup_training_cli_parser(
        description="CBraMod Manifest-Based Pipeline for Embedding Extraction & Probe Training"
    )

    # Checkpoint & Cache Controls
    cache_group = parser.add_argument_group("Cache Controls")
    cache_group.add_argument("--cache-dir", type=str, default=None, help="Directory for cached embeddings")
    cache_group.add_argument("--train-cache-name", type=str, default="cached_train_embeddings.pt", help="Filename for cached training embeddings")
    cache_group.add_argument("--val-cache-name", type=str, default="cached_val_embeddings.pt", help="Filename for cached validation embeddings")
    cache_group.add_argument("--force-extract", action="store_true", help="Force re-extraction of backbone embeddings")

    # Run Options
    run_group = parser.add_argument_group("Run Options")
    run_group.add_argument("--enable-sgkf", action="store_true", help="Enable stratified group k-fmax old cross-validation (overrides train/val split)")
    run_group.add_argument("--sgkf-folds", type=int, default=5, help="Number of folds for stratified group k-fold CV (default: 5)")

    # Logging Controls
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-filename", type=str, default=Path(__file__).stem + ".log", help="Filename for pipeline log output")

    args = parser.parse_args()
    args.use_amp = not args.no_amp
    return args

# =====================================================================
# 2. EMBEDDING EXTRACTION ENGINE WITH STAGE FILTERING & SUBJECT ID TRACKING
# =====================================================================

class EmbeddingManager:
    """Manages feature extraction from manifest .npy files with optional stage filtering."""
    def __init__(self, config: argparse.Namespace, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ProbeTrainer(CBraModTrainer):
    """Trains head on window instances while evaluating and tuning thresholds on subject-level pooled scores."""
    def __init__(self, config: argparse.Namespace, logger: logging.Logger):
        super().__init__(config, logger)

    def train(self, train_path: Path, val_path: Path) -> dict:
        """Main training loop that handles both fixed split and optional SGKF cross-validation."""
        if self.config.enable_sgkf:
            return self.train_cross_validation(train_path, val_path)
        else:
            return self.train_fixed_split(train_path, val_path)
    
    def train_fixed_split(self, train_path: Path, val_path: Path) -> dict:
        """Trains the probe head on a fixed train/val split and evaluates subject-level pooling metrics."""
        self.logger.info("Loading cached feature tensors into RAM...")
        train_data = torch.load(train_path, map_location="cpu", weights_only=True)
        val_data = torch.load(val_path, map_location="cpu", weights_only=True)

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

        if self.config.head_type == "linear":
            head = LinearProbeHead(
                num_patches=self.config.num_patches,
                emb_dim=self.config.cbra_dim,
                num_classes=self.config.num_classes
            )
        elif self.config.head_type == "mlp":
            head = MLPProbeHead(
                num_patches=self.config.num_patches,
                emb_dim=self.config.cbra_dim,
                hidden_dim=self.config.head_dim,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout
            )

        head.to(self.device)

        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config.head_lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr)

        best_primary_metrics = {}
        best_primary_f1 = 0.0
        best_thresholds = {}
        patience_counter = 0
        best_model_path = Path(self.config.cache_dir) / self.config.checkpoint_filename

        self.logger.info(
            f"Starting Probe Training ({self.config.epochs} Epochs Max | Batch Size: {self.config.batch_size} | "
            f"Head Type: {self.config.head_type} | "
            f"Imbalance: {self.config.imbalance_strategy} | Pooling: {self.config.primary_pooling} | "
            f"Head LR: {self.config.head_lr} | Weight Decay: {self.config.weight_decay})"
        )
        self.logger.info("=" * 125)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            current_lr = scheduler.get_last_lr()[0]

            # 1. Window-Level Training Loop
            head.train()
            train_loss, train_correct, train_total_samples = 0.0, 0, 0
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
                train_total_samples += len(y_b)

            train_loss /= train_total_samples
            train_acc = (train_correct / train_total_samples) * 100.0

            # 2. Window-Level Inference on Validation
            head.eval()
            val_loss, val_correct, val_total_samples = 0.0, 0, 0
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b = x_b.to(self.device, non_blocking=True).float()
                    y_b = y_b.to(self.device, non_blocking=True)

                    out = head(x_b)
                    loss = criterion(out, y_b)
                    probs = torch.softmax(out, dim=1)

                    val_loss += loss.item() * len(y_b)
                    val_correct += (out.argmax(dim=1) == y_b).sum().item()
                    val_total_samples += len(y_b)

                    val_preds.append(out.argmax(dim=1).cpu().numpy())
                    val_targets.append(y_b.cpu().numpy())
                    val_probs.append(probs.cpu().numpy())

            val_loss /= val_total_samples
            val_acc = (val_correct / val_total_samples) * 100.0

            val_targets = np.concatenate(val_targets)
            val_probs = np.concatenate(val_probs)

            # 3. Fast Subject-Level Multi-Strategy Pooling & Threshold Calibration
            pooling_results = self.evaluate_subject_pooling(
                val_probs=val_probs,
                val_targets=val_targets,
                val_subject_ids=val_subject_ids
            )
            
            primary_metrics = pooling_results[self.config.primary_pooling]
            primary_f1 = primary_metrics["subject_macro_f1"]
            primary_t = primary_metrics["optimal_threshold"]
            primary_acc = primary_metrics["subject_accuracy"]
            primary_auc = primary_metrics["roc_auc"]

            scheduler.step()
            elapsed = time.time() - t0

            # Format log string showing window loss + subject-level pooled metrics across strategies
            log_str = (
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.2f}s) | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Subj Acc: {primary_acc*100:.2f}% | "
                f"Subj F1 ({self.config.primary_pooling}@{primary_t:.2f}): {primary_f1:.4f} | "
                f"Subj AUC: {primary_auc:.4f}"
            )

            # Model Selection & Checkpointing based on Primary Subject-Level Macro F1
            if primary_f1 > best_primary_f1:
                best_primary_f1 = primary_f1
                best_primary_metrics = primary_metrics
                best_thresholds = {strat: res["optimal_threshold"] for strat, res in pooling_results.items()}

                patience_counter = 0
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
                log_str += f" | EarlyStop: {patience_counter}/{self.config.patience}"

            self.logger.info(log_str)

            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.logger.info("=" * 125)
        self.logger.info(
            f"Training Complete. Best Validation Subject Macro F1 ({self.config.primary_pooling}): {best_primary_f1:.4f}"
        )
        self.logger.info(f"Calibrated Strategy Thresholds: {best_thresholds}")

        return best_primary_metrics

    def train_cross_validation(self, train_cache_path: Path, val_cache_path: Path) -> dict:
        """Executes Stratified Group K-Fold Cross-Validation on subject-level splits."""
        # Load and concatenate both train and val caches to form a unified pool for SGKF
        self.logger.info("Loading cached embeddings into memory for SGKF splitting...")
        train_data = torch.load(train_cache_path, map_location="cpu", weights_only=True)
        val_data = torch.load(val_cache_path, map_location="cpu", weights_only=True)

        all_feats = torch.cat([train_data["feats"], val_data["feats"]], dim=0)
        all_labels = torch.cat([train_data["labels"], val_data["labels"]], dim=0)
    
        # Combine subject IDs safely
        train_sids = train_data.get("subject_ids", [])
        val_sids = val_data.get("subject_ids", [])
        all_subject_ids = np.array(train_sids + val_sids)

        # Build unique subject-to-label mapping for StratifiedGroupKFold
        unique_sids = np.unique(all_subject_ids)
        subject_labels = {}
        for sid in unique_sids:
            mask = (all_subject_ids == sid)
            # Take the majority label or first window label as the subject label representation
            subject_labels[sid] = int(all_labels[mask][0].item())

        unique_labels = np.array([subject_labels[s] for s in unique_sids])

        # Execute k-Fold Stratified Group K-Fold
        sgkf = StratifiedGroupKFold(n_splits=self.config.sgkf_folds, shuffle=True, random_state=self.config.seed)
    
        fold_results = []
    
        self.logger.info(f"=" * 100)
        self.logger.info(f"STARTING {self.config.sgkf_folds}-FOLD STRATIFIED GROUP K-FOLD CROSS-VALIDATION ACROSS {len(unique_sids)} TOTAL SUBJECTS")
        self.logger.info(f"=" * 100)

        # Save the original checkpoint filename
        checkpoint_path = Path(self.config.checkpoint_filename)

        for fold, (train_subj_idx, val_subj_idx) in enumerate(
            sgkf.split(unique_sids, unique_labels, groups=unique_sids)):
            self.logger.info(f"\n--- Fold [{fold+1}/{self.config.sgkf_folds}] ---")
            train_sids_fold = set(unique_sids[train_subj_idx])
            val_sids_fold = set(unique_sids[val_subj_idx])

            train_mask = np.isin(all_subject_ids, list(train_sids_fold))
            val_mask = np.isin(all_subject_ids, list(val_sids_fold))

            self.sanity_check(all_feats, all_subject_ids, fold, train_sids_fold, val_sids_fold, train_mask, val_mask)

            # Save temporary fold-specific cache paths to leverage existing ProbeTrainer class directly
            cache_dir = Path(self.config.cache_dir)
            fold_train_cache = cache_dir / f"fold_{fold+1}_train.pt"
            fold_val_cache = cache_dir / f"fold_{fold+1}_val.pt"

            torch.save({
                "feats": all_feats[train_mask],
                "labels": all_labels[train_mask],
                "subject_ids": all_subject_ids[train_mask].tolist()
            }, fold_train_cache)

            torch.save({
                "feats": all_feats[val_mask],
                "labels": all_labels[val_mask],
                "subject_ids": all_subject_ids[val_mask].tolist()
            }, fold_val_cache)

            # Update checkpoint filename per fold to prevent overwriting
            self.config.checkpoint_filename = checkpoint_path.with_stem(f"{checkpoint_path.stem}_fold_{fold+1}")

            results = self.train_fixed_split(fold_train_cache, fold_val_cache)
            fold_results.append(results)

        # Aggregate OOF Summary Statistics
        stats = {}
        for metric in fold_results[0].keys():
            metric_values = [res[metric] for res in fold_results]
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            stats[metric] = {"mean": mean_val, "std": std_val}

        self.logger.info(f"=" * 100)
        self.logger.info(f"{self.config.sgkf_folds}-FOLD CROSS-VALIDATION COMPLETE")
        for metric, value in stats.items():
            self.logger.info(f"{metric}: {value["mean"]:.4f} +/- {value["std"]:.4f}")
        self.logger.info(f"=" * 100)

        return stats

    def sanity_check(self, all_feats, all_subject_ids, fold, train_sids_fold, val_sids_fold, train_mask, val_mask):
        # =====================================================================
        # STRICT LEAKAGE & PARTITION DEBUG CHECKS
        # =====================================================================
        # 1. Check Subject ID set intersection
        id_intersection = train_sids_fold.intersection(val_sids_fold)
        assert len(id_intersection) == 0, (
            f"[CRITICAL LEAKAGE] Fold {fold+1}: Found {len(id_intersection)} overlapping subject IDs in group split! "
            f"Leaked IDs: {id_intersection}"
        )

        # 2. Check window mask mutual exclusivity (no window in both train and val)
        mask_overlap = np.sum(train_mask & val_mask)
        assert mask_overlap == 0, (
            f"[CRITICAL LEAKAGE] Fold {fold+1}: {mask_overlap} window samples assigned to BOTH train and val splits!"
        )

        # 3. Check partition completeness (no windows silently dropped)
        dropped_windows = np.sum(~(train_mask | val_mask))
        assert dropped_windows == 0, (
            f"[DATA LOSS] Fold {fold+1}: {dropped_windows} windows unmapped during string/type masking!"
        )

        # 4. Verify actual subject IDs extracted from filtered feature arrays
        actual_train_subjs = set(all_subject_ids[train_mask])
        actual_val_subjs = set(all_subject_ids[val_mask])
        array_intersection = actual_train_subjs.intersection(actual_val_subjs)
        assert len(array_intersection) == 0, (
            f"[CRITICAL LEAKAGE] Fold {fold+1}: Found {len(array_intersection)} overlapping subjects in feature arrays! "
            f"Overlapping: {array_intersection}"
        )

        self.logger.info(
            f"✓ [Leak Checks Passed] Fold {fold+1} | "
            f"Train: {len(actual_train_subjs)} subjs ({np.sum(train_mask):,} windows) | "
            f"Val: {len(actual_val_subjs)} subjs ({np.sum(val_mask):,} windows) | "
            f"Total Retained: {len(all_feats):,} windows"
        )
        # =====================================================================


# =====================================================================
# 4. PIPELINE ORCHESTRATOR
# =====================================================================

def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cache_dir / args.log_filename)

    train_cache_path = cache_dir / args.train_cache_name
    val_cache_path = cache_dir / args.val_cache_name
    cache_exists = train_cache_path.exists() and val_cache_path.exists()

    if not cache_exists or args.force_extract:
        logger.info("Cached embeddings not found or --force-extract specified. Initializing Feature Extraction Phase...")

        if not args.train_manifest or not args.val_manifest:
            logger.error("Missing manifest paths! Please provide --train-manifest and --val-manifest to extract features.")
            sys.exit(1)

        train_manifest_path = Path(args.train_manifest)
        val_manifest_path = Path(args.val_manifest)

        extractor_mgr = EmbeddingManager(args, logger)
        extractor_mgr.extract_and_cache(train_manifest_path, train_cache_path, split_name="train")
        extractor_mgr.extract_and_cache(val_manifest_path, val_cache_path, split_name="val")
    else:
        logger.info(f"Found existing cached feature files at '{cache_dir}'. Skipping .npy extraction phase.")

    trainer = ProbeTrainer(args, logger)
    trainer.train(train_cache_path, val_cache_path)


if __name__ == "__main__":
    main()
