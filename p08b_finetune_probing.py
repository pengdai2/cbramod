"""
Production CLI Pipeline for CBraMod Probe Training & Subject-Level Multi-Strategy Pooling Evaluation.

Reads backbone embeddings from a single master cache -- built once, for the WHOLE cohort (train/val/
test all together), by the standalone p08a_extract_features.py -- rather than extracting them itself.

--train-manifest/--val-manifest (p03's per-split CSVs) are read for their subject_id column only, and
mean something slightly different depending on mode:
  - Fixed split (default): the two lists ARE the train/val subject split. CachedFeatureSubjectDataset's
    subject filter carves each view directly out of the master cache.
  - --enable-sgkf: their UNION defines which subjects are ELIGIBLE for cross-validation at all --
    StratifiedGroupKFold then draws its own train/val partition per fold from that pool. This is what
    keeps held-out test subjects (present in the master cache, since it covers everyone) out of every
    fold's train AND val partition -- --train-manifest/--val-manifest are still required in this mode,
    just for a different purpose than fixing a split.

Either way, there's no separate extraction pass per split and no temporary per-fold cache files.

Usage:
  # 1. First, build the master cache once (see p08a_extract_features.py):
  python p08a_extract_features.py \
      --master-manifest /data/eeg_study/master_manifest.csv \
      --data-dir /data/eeg_study/npy_files --filter-stage N2,N3 \
      --cache-dir /data/eeg_study/cache

  # 2. Train against a fixed train/val split, drawn from that cache:
  python p08b_finetune_probing.py \
      --cache-dir /data/eeg_study/cache \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --head-lr 0.0003 --epochs 40 --primary-pooling p85_score

  # 3. Stratified Group K-Fold CV over the train+val subject pool (test subjects, present in the
  #    master cache, are excluded from every fold -- see train_cross_validation()'s docstring):
  python p08b_finetune_probing.py \
      --cache-dir /data/eeg_study/cache \
      --train-manifest /data/eeg_study/train_manifest.csv \
      --val-manifest /data/eeg_study/val_manifest.csv \
      --enable-sgkf --sgkf-folds 5
"""

import argparse
import logging
from pathlib import Path
from typing import List
import sys
import time

import numpy as np
import pandas as pd
from cbramod_utils import setup_logger
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

from cbramod_common import (
    CachedFeatureSubjectDataset,
    CBraModTrainer,
    LinearProbeHead,
    MLPProbeHead,
    add_log_filename_argument,
    flatten_cached_feature_dataset,
    is_checkpoint_improvement,
    load_subject_ids,
    seed_everything,
    setup_cache_cli_parser,
    setup_data_loader_and_criterion,
    setup_training_cli_parser,
)


# =====================================================================
# 1. CONFIGURATION & CLI PARSER
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for the CBraMod probe-training pipeline."""
    parser = setup_training_cli_parser(
        description="CBraMod Probe Training & Subject-Level Multi-Strategy Pooling Evaluation "
                     "(reads a master cache built by p08a_extract_features.py)"
    )

    setup_cache_cli_parser(parser)

    # Run Options
    run_group = parser.add_argument_group("Run Options")
    run_group.add_argument("--enable-sgkf", action="store_true", help="Enable stratified group k-fold cross-validation (overrides train/val split)")
    run_group.add_argument("--sgkf-folds", type=int, default=5, help="Number of folds for stratified group k-fold CV (default: 5)")

    add_log_filename_argument(parser, __file__)

    args = parser.parse_args()
    args.use_amp = not args.no_amp
    return args


# =====================================================================
# 2. TRAINER ENGINE WITH WINDOW TRAINING & FAST SUBJECT POOLING VAL
# =====================================================================

class ProbeTrainer(CBraModTrainer):
    """Trains head on window instances while evaluating and tuning thresholds on subject-level pooled scores."""
    def __init__(self, config: argparse.Namespace, logger: logging.Logger):
        super().__init__(config, logger)

    def train(self, master_cache_path: Path) -> dict:
        """Main training loop that handles both fixed split and optional SGKF cross-validation."""
        if self.config.enable_sgkf:
            return self.train_cross_validation(master_cache_path)
        else:
            train_subject_ids = load_subject_ids(Path(self.config.train_manifest))
            val_subject_ids = load_subject_ids(Path(self.config.val_manifest))
            return self.train_fixed_split(master_cache_path, train_subject_ids, val_subject_ids)

    def train_fixed_split(
        self, master_cache_path: Path, train_subject_ids: List[str], val_subject_ids: List[str]
    ) -> dict:
        """
        Trains the probe head on a fixed train/val subject split and evaluates subject-level pooling
        metrics. Both splits are carved out of the SAME master cache via CachedFeatureSubjectDataset's
        subject filter -- no separate per-split extraction, and (called per-fold from
        train_cross_validation) no temporary per-fold cache files either.
        """
        self.logger.info(f"Loading cached feature tensors into RAM from {master_cache_path}...")
        train_view = CachedFeatureSubjectDataset(master_cache_path, filter_subject=train_subject_ids)
        val_view = CachedFeatureSubjectDataset(master_cache_path, filter_subject=val_subject_ids)

        # Same invariant the old CV-only sanity_check() checked, now enforced for EVERY fixed-split
        # call (plain train/val included, not just CV folds) since nothing here is CV-specific.
        leaked = set(train_view.unique_subjects) & set(val_view.unique_subjects)
        assert not leaked, f"[CRITICAL LEAKAGE] {len(leaked)} subject(s) in both train and val: {leaked}"

        train_feats, train_labels, _, _, _ = flatten_cached_feature_dataset(train_view)
        val_feats, val_labels, val_subject_ids_flat, _, _ = flatten_cached_feature_dataset(val_view)

        self.logger.info(
            f"✓ [Leak Check Passed] Train: {len(train_view.unique_subjects)} subjs ({len(train_feats):,} windows) | "
            f"Val: {len(val_view.unique_subjects)} subjs ({len(val_feats):,} windows)"
        )

        train_ds = TensorDataset(train_feats, train_labels)
        val_ds = TensorDataset(val_feats, val_labels)

        train_loader, criterion = setup_data_loader_and_criterion(
            dataset=train_ds,
            labels=train_labels.numpy(),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            imbalance_strategy=self.config.imbalance_strategy,
            device=self.device,
            logger=self.logger
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

        # val_subject_ids_flat comes straight from flatten_cached_feature_dataset() above -- always
        # present (CachedFeatureSubjectDataset requires subject_ids to load at all), so no fallback needed.
        val_subject_ids = val_subject_ids_flat

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
        best_primary_auc = 0.0
        best_thresholds = {}
        patience_counter = 0
        best_model_path = Path(self.config.checkpoint_dir) / self.config.checkpoint_filename

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

            # Model Selection & Checkpointing: strict Pareto criterion on Primary Subject-Level Macro
            # F1 AND AUC -- neither may regress, at least one must strictly improve. See
            # is_checkpoint_improvement()'s docstring in cbramod_common.py for the full rationale
            # (guards against both an F1-only check's plateau-while-AUC-climbs blind spot, and
            # against crediting an F1 uptick that came at a real AUC cost).
            if is_checkpoint_improvement(primary_f1, primary_auc, best_primary_f1, best_primary_auc):
                best_primary_f1 = primary_f1
                best_primary_auc = primary_auc
                best_primary_metrics = primary_metrics
                best_thresholds = {strat: res["optimal_threshold"] for strat, res in pooling_results.items()}

                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": head.state_dict(),
                        # Explicit architecture metadata -- saved so any downstream consumer of this
                        # checkpoint (e.g. p13_attention_mil_pooling.py's frozen-probe loader) can
                        # reconstruct the exact right head without having to infer it from state_dict
                        # key/shape patterns or trust a CLI flag to happen to match what this
                        # checkpoint was actually trained with.
                        "head_type": self.config.head_type,
                        "head_dim": self.config.head_dim,
                        "num_patches": self.config.num_patches,
                        "cbra_dim": self.config.cbra_dim,
                        "num_classes": self.config.num_classes,
                        "best_macro_f1": best_primary_f1,
                        "best_auc": best_primary_auc,
                        "primary_pooling": self.config.primary_pooling,
                        "top_percentile": self.config.top_percentile,
                        "t_window": self.config.t_window,
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
            f"Training Complete. Best Validation Subject Macro F1 ({self.config.primary_pooling}): "
            f"{best_primary_f1:.4f} | Best AUC: {best_primary_auc:.4f}"
        )
        self.logger.info(f"Calibrated Strategy Thresholds: {best_thresholds}")

        return best_primary_metrics

    def train_cross_validation(self, master_cache_path: Path) -> dict:
        """
        Executes Stratified Group K-Fold Cross-Validation on subject-level splits, all drawn from the
        SAME master cache -- no concatenation of separate train/val caches, and no temporary per-fold
        cache files (train_fixed_split() carves each fold's train/val subject subset directly out of
        master_cache_path via CachedFeatureSubjectDataset's filter).

        The master cache covers the WHOLE cohort (p03's master_manifest.csv includes train/val/test),
        so the SGKF pool must be explicitly restricted to subjects in --train-manifest/--val-manifest
        before splitting -- otherwise a held-out test subject could land in some fold's train OR val
        partition, and the test set stops being genuinely held out. --train-manifest/--val-manifest
        are repurposed here as "which subjects are eligible for CV at all" rather than a fixed split.
        """
        if not self.config.train_manifest or not self.config.val_manifest:
            raise ValueError(
                "--enable-sgkf requires --train-manifest and --val-manifest -- not to fix a split, but "
                "to determine which subjects are eligible for cross-validation at all (their union), "
                "explicitly excluding whatever's held out as test in the master cache."
            )
        eligible_subject_ids = set(load_subject_ids(Path(self.config.train_manifest))) | \
            set(load_subject_ids(Path(self.config.val_manifest)))

        self.logger.info(f"Loading master cache from {master_cache_path} to build SGKF splits...")
        master_data = torch.load(master_cache_path, map_location="cpu", weights_only=True)
        missing_keys = [k for k in ("subject_ids", "labels", "stages", "indices") if k not in master_data]
        if missing_keys:
            raise KeyError(
                f"Master cache '{master_cache_path}' is missing key(s) {missing_keys} -- re-run "
                "p08a_extract_features.py to regenerate it."
            )

        all_subject_ids = np.array(master_data["subject_ids"])
        all_labels = master_data["labels"]

        # Restrict to the eligible pool BEFORE computing unique_sids -- excluded subjects (e.g. test)
        # must never enter sgkf.split() at all, not just get filtered out of the resulting folds.
        cv_pool_mask = np.isin(all_subject_ids, list(eligible_subject_ids))
        excluded_subjects = np.unique(all_subject_ids[~cv_pool_mask])
        if len(excluded_subjects) > 0:
            self.logger.info(
                f"Excluding {len(excluded_subjects)} subject(s) present in the master cache but not in "
                f"--train-manifest/--val-manifest (e.g. held-out test) from the SGKF pool."
            )
        all_subject_ids = all_subject_ids[cv_pool_mask]
        all_labels = all_labels[cv_pool_mask]

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
            train_sids_fold = list(unique_sids[train_subj_idx])
            val_sids_fold = list(unique_sids[val_subj_idx])

            # Update checkpoint filename per fold to prevent overwriting
            self.config.checkpoint_filename = checkpoint_path.with_stem(f"{checkpoint_path.stem}_fold_{fold+1}")

            # train_fixed_split() re-runs its own leakage assertion on top of this -- the SGKF split
            # itself guarantees disjoint subject sets by construction, but checking again at the point
            # where train/val views actually get built is what catches a REAL bug, not just a
            # theoretical one.
            results = self.train_fixed_split(master_cache_path, train_sids_fold, val_sids_fold)
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
            self.logger.info(f"{metric}: {value['mean']:.4f} +/- {value['std']:.4f}")
        self.logger.info(f"=" * 100)

        return stats


# =====================================================================
# 3. PIPELINE ORCHESTRATOR
# =====================================================================

def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cache_dir / args.log_filename)

    master_cache_path = cache_dir / args.master_cache_name
    if not master_cache_path.exists():
        logger.error(
            f"Master cache not found at '{master_cache_path}'. Run p08a_extract_features.py first "
            "to build it (once, for the whole cohort) -- this script only trains against an "
            "already-extracted cache, it doesn't extract features itself."
        )
        sys.exit(1)

    trainer = ProbeTrainer(args, logger)
    trainer.train(master_cache_path)


if __name__ == "__main__":
    main()
