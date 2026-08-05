"""
Production End-to-End Fine-Tuning Pipeline for CBraMod.
Uses PANSleepEEGDataset's native stage filtering, CBraModE2EClassifier,
class imbalance handling (WeightedRandomSampler, class loss weighting, or none),
flexible backbone unfreezing, warm-start probing head initialization, and 
vectorized subject-level validation pooling.
"""

import argparse
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared architectures and utilities without code duplication
from cbramod_common import CBraModE2EClassifier, LinearProbeHead, PANSleepEEGDataset, compute_pooled_scores, setup_data_loader_and_criterion
from cbramod_utils import seed_everything, setup_logger


# =====================================================================
# 1. UNFREEZING & CHECKPOINT UTILITIES
# =====================================================================

def configure_backbone_unfreezing(
    model: CBraModE2EClassifier,
    unfreeze_mode: str = "full",
    unfreeze_last_n: int = 2,
    logger: Optional[logging.Logger] = None
) -> None:
    """Configures parameter trainable status for partial or full fine-tuning."""
    backbone = getattr(model, "backbone", model)
    head = getattr(model, "head", getattr(model, "classifier", None))

    # Freeze all backbone parameters by default
    for param in backbone.parameters():
        param.requires_grad = False

    # Ensure classification head is trainable
    if head is not None:
        for param in head.parameters():
            param.requires_grad = True

    if unfreeze_mode == "head_only":
        msg = "Unfreeze Mode [HEAD_ONLY]: Entire CBraMod backbone is frozen."
    elif unfreeze_mode == "full":
        for param in backbone.parameters():
            param.requires_grad = True
        msg = "Unfreeze Mode [FULL]: Entire CBraMod backbone is unfrozen."
    elif unfreeze_mode == "partial":
        children = list(backbone.named_children())
        unfrozen_modules = children[-unfreeze_last_n:] if len(children) >= unfreeze_last_n else children
        
        for name, module in unfrozen_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        unfrozen_names = [name for name, _ in unfrozen_modules]
        msg = f"Unfreeze Mode [PARTIAL]: Unfrozen last {len(unfrozen_names)} backbone blocks -> {unfrozen_names}"
    else:
        raise ValueError(f"Invalid unfreeze_mode: {unfreeze_mode}")

    if logger:
        logger.info(msg)
    else:
        print(msg)


def load_probing_head_checkpoint(
    model: CBraModE2EClassifier,
    checkpoint_path: Path,
    logger: logging.Logger
) -> None:
    """Loads state dict from a linear probing checkpoint into the classifier head."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Probing head checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading pre-trained linear probing head from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = ckpt.get("model_state_dict", ckpt.get("head_state_dict", ckpt))
    head_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("head.") or k.startswith("classifier."):
            clean_k = k.replace("head.", "").replace("classifier.", "")
            head_state_dict[clean_k] = v
        elif not k.startswith("backbone."):
            head_state_dict[k] = v

    target_head = getattr(model, "head", getattr(model, "classifier", None))
    if target_head is not None:
        target_head.load_state_dict(head_state_dict, strict=True)
        logger.info("✓ Linear probe head weights successfully warm-started.")
    else:
        logger.warning("Could not identify classification head on model to load checkpoint weights.")


# =====================================================================
# 2. E2E FINE-TUNING TRAINER
# =====================================================================

class EndToEndTrainer:
    """Manages full/partial E2E backpropagation and subject-level threshold calibration."""
    def __init__(self, config: argparse.Namespace, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subject_index_map: Optional[Dict[str, np.ndarray]] = None
        self.subject_labels_map: Optional[np.ndarray] = None

    def _prepare_subject_indices(self, val_subject_ids: List[str], val_targets: np.ndarray) -> None:
        """Pre-computes window index maps for subject evaluation once to avoid O(S*N) lookups per epoch."""
        val_subject_ids_np = np.array(val_subject_ids)
        unique_subjects, first_indices = np.unique(val_subject_ids_np, return_index=True)
        
        self.subject_index_map = {
            subj: np.where(val_subject_ids_np == subj)[0] for subj in unique_subjects
        }
        self.subject_labels_map = val_targets[first_indices]

    def _evaluate_subject_pooling(self, val_probs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Executes subject-level aggregation and vectorized threshold tuning across pooling strategies."""
        strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]
        subject_data = {strat: [] for strat in strategies}

        for subj, idx in self.subject_index_map.items():
            subj_probs = val_probs[idx]

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

        subject_labels = self.subject_labels_map
        results = {}
        thresholds = np.linspace(0.01, 0.99, 99)

        for strat in strategies:
            scores = np.array(subject_data[strat])
            if self.config.num_classes == 2:
                preds_matrix = (scores[:, None] >= thresholds[None, :]).astype(int)
                pos_mask = (subject_labels == 1)[:, None]
                neg_mask = (subject_labels == 0)[:, None]
                
                tp = np.sum(preds_matrix & pos_mask, axis=0)
                fp = np.sum(preds_matrix & neg_mask, axis=0)
                fn = np.sum((~preds_matrix) & pos_mask, axis=0)
                tn = np.sum((~preds_matrix) & neg_mask, axis=0)

                f1_pos = np.where((2 * tp + fp + fn) > 0, (2 * tp) / (2 * tp + fp + fn), 0.0)
                f1_neg = np.where((2 * tn + fp + fn) > 0, (2 * tn) / (2 * tn + fp + fn), 0.0)
                macro_f1s = (f1_pos + f1_neg) / 2.0

                best_idx = int(np.argmax(macro_f1s))
                best_f1 = float(macro_f1s[best_idx])
                best_t = float(thresholds[best_idx])

                final_preds = preds_matrix[:, best_idx]
                acc = accuracy_score(subject_labels, final_preds)
                roc_auc = roc_auc_score(subject_labels, scores) if len(np.unique(subject_labels)) > 1 else 0.5

                results[strat] = {
                    "subject_macro_f1": best_f1,
                    "optimal_threshold": best_t,
                    "subject_accuracy": acc,
                    "roc_auc": roc_auc
                }
            else:
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

    def fit(self, train_manifest: Path, val_manifest: Path) -> float:
        allowed_stages = [s.strip() for s in self.config.filter_stage.split(",") if s.strip()] if self.config.stages else None

        # 1. Instantiate Datasets using built-in stage filtering
        train_ds = PANSleepEEGDataset(
            manifest_csv=train_manifest,
            data_dir=self.config.data_dir,
            filter_stage=allowed_stages
        )
        val_ds = PANSleepEEGDataset(
            manifest_csv=val_manifest,
            data_dir=self.config.data_dir,
            filter_stage=allowed_stages
        )

        # 2. Configure Strategy-Based Data Loaders and Loss Function
        train_loader, criterion = setup_data_loader_and_criterion(
            train_ds=train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            imbalance_strategy=self.config.imbalance_strategy,
            device=self.device,
            logger=self.logger
        )

        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

        # Extract Subject IDs for validation pooling
        df_val = pd.read_csv(val_manifest)
        if allowed_stages and "stage" in df_val.columns:
            df_val = df_val[df_val["stage"].astype(str).str.upper().isin([s.upper() for s in allowed_stages])]

        val_subject_ids = (
            df_val["subject_id"].astype(str).tolist()
            if "subject_id" in df_val.columns
            else [Path(p).stem for p in df_val["npy_path"]]
        )
        val_targets = np.array([sample[1] for sample in val_ds.samples])

        # Pre-index subjects for fast validation evaluation
        self._prepare_subject_indices(val_subject_ids, val_targets)

        # 3. Build Model Architecture
        model = CBraModE2EClassifier(
            num_channels=self.config.num_channels,
            num_classes=self.config.num_classes
        ).to(self.device)

        # Load Probing Head Checkpoint if specified
        if self.config.probe_head_ckpt:
            load_probing_head_checkpoint(model, Path(self.config.probe_head_ckpt), self.logger)

        # Configure Backbone Unfreezing Mode
        configure_backbone_unfreezing(
            model=model,
            unfreeze_mode=self.config.unfreeze_mode,
            unfreeze_last_n=self.config.unfreeze_last_n,
            logger=self.logger
        )

        # Configure Parameter Groups with Differential Learning Rates
        backbone = getattr(model, "backbone", model)
        head = getattr(model, "head", getattr(model, "classifier", None))

        backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
        head_trainable = [p for p in head.parameters() if p.requires_grad] if head else []

        optimizer_grouped_parameters = []
        if backbone_trainable:
            optimizer_grouped_parameters.append({"params": backbone_trainable, "lr": self.config.lr_backbone})
        if head_trainable:
            optimizer_grouped_parameters.append({"params": head_trainable, "lr": self.config.lr_head})

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr)
        scaler = torch.amp.GradScaler(device="cuda", enabled=self.config.use_amp)

        best_primary_f1 = 0.0
        patience_counter = 0
        best_model_path = Path(self.config.checkpoint_dir) / "cbramod_e2e_best.pt"

        self.logger.info(
            f"Starting E2E Training ({self.config.epochs} Epochs | Batch Size: {self.config.batch_size} | "
            f"Strategy: {self.config.imbalance_strategy} | Backbone LR: {self.config.lr_backbone} | Head LR: {self.config.lr_head})"
        )
        self.logger.info("=" * 125)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            
            # Training Phase
            model.train()
            train_loss, train_correct, total_train_samples = 0.0, 0, 0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]", leave=False)
            for step, batch in enumerate(pbar):
                x_b, y_b = batch[0], batch[1]
                x_b = x_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                    loss = loss / self.config.grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * self.config.grad_accum_steps * len(y_b)
                train_correct += (logits.argmax(dim=1) == y_b).sum().item()
                total_train_samples += len(y_b)

            train_loss /= total_train_samples
            train_acc = (train_correct / total_train_samples) * 100.0

            # Validation Inference Phase
            model.eval()
            val_loss = 0.0
            val_probs = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Val]", leave=False):
                    x_b, y_b = batch[0], batch[1]
                    x_b = x_b.to(self.device, non_blocking=True)
                    y_b = y_b.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
                        logits = model(x_b)
                        loss = criterion(logits, y_b)
                        probs = torch.softmax(logits, dim=1)

                    val_loss += loss.item() * len(y_b)
                    val_probs.append(probs.cpu().numpy())

            val_loss /= len(val_ds)
            val_probs = np.concatenate(val_probs, axis=0)

            # Subject-Level Multi-Strategy Pooling Evaluation
            pooling_results = self._evaluate_subject_pooling(val_probs)
            primary_metrics = pooling_results[self.config.primary_pooling]
            primary_f1 = primary_metrics["subject_macro_f1"]
            primary_t = primary_metrics["optimal_threshold"]
            primary_acc = primary_metrics["subject_accuracy"]

            scheduler.step()
            elapsed = time.time() - t0

            log_str = (
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Subj Acc: {primary_acc*100:.2f}% | "
                f"Subj F1 ({self.config.primary_pooling}@{primary_t:.2f}): {primary_f1:.4f}"
            )

            if primary_f1 > best_primary_f1:
                best_primary_f1 = primary_f1
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_macro_f1": best_primary_f1,
                        "primary_pooling": self.config.primary_pooling,
                        "optimal_thresholds": {k: v["optimal_threshold"] for k, v in pooling_results.items()},
                    },
                    best_model_path,
                )
                log_str += " --> [BEST MODEL SAVED]"
            else:
                patience_counter += 1
                log_str += f" | EarlyStop: {patience_counter}/{self.config.patience}"

            self.logger.info(log_str)

            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self.logger.info("=" * 125)
        self.logger.info(f"E2E Fine-Tuning Complete. Best Subject Macro F1: {best_primary_f1:.4f}")
        return best_primary_f1


# =====================================================================
# 4. CLI ORCHESTRATOR
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CBraMod End-to-End Fine-Tuning Pipeline with Imbalance Strategy Controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Manifest & Data Sources
    data_group = parser.add_argument_group("Data & Checkpoints")
    data_group.add_argument("--train-manifest", type=str, required=True, help="Path to training manifest CSV")
    data_group.add_argument("--val-manifest", type=str, required=True, help="Path to validation manifest CSV")
    data_group.add_argument("--data-dir", type=str, default=None, help="Root directory for .npy EEG files")
    data_group.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save fine-tuned checkpoints")
    data_group.add_argument("--probe-head-ckpt", type=str, default=None, help="Optional path to warm-start linear probe head checkpoint")

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

    # Unfreezing Controls
    unfreeze_group = parser.add_argument_group("Backbone Unfreezing Controls")
    unfreeze_group.add_argument(
        "--unfreeze-mode", type=str, default="partial", choices=["head_only", "partial", "full"],
        help="Unfreezing strategy for backbone parameters"
    )
    unfreeze_group.add_argument("--unfreeze-last-n", type=int, default=2, help="Number of top backbone submodules to unfreeze in 'partial' mode")

    # Hyperparameters
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument("--epochs", type=int, default=20, help="Maximum fine-tuning epochs")
    hp_group.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    hp_group.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for trainable backbone parameters")
    hp_group.add_argument("--lr-head", type=float, default=3e-4, help="Learning rate for classification head")
    hp_group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing scheduler")
    hp_group.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    hp_group.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")

    # System & Validation Options
    sys_group = parser.add_argument_group("System & Pooling Options")
    sys_group.add_argument("--num-channels", type=int, default=64, help="EEG Channel count")
    sys_group.add_argument("--sfreq", type=float, default=200.0, help="EEG sampling frequency")
    sys_group.add_argument("--num-classes", type=int, default=2, help="Number of target classes")
    sys_group.add_argument("--primary-pooling", type=str, default="p85_score", choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"])
    sys_group.add_argument("--top-percentile", type=float, default=0.10, help="Top percentile ratio for top-K pooling methods")
    sys_group.add_argument("--t-window", type=float, default=0.60, help="Window threshold for pathology burden ratio")
    sys_group.add_argument("--num-workers", type=int, default=4, help="DataLoader CPU workers")
    sys_group.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    sys_group.add_argument("--seed", type=int, default=42, help="Random seed")
    sys_group.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")

    args = parser.parse_args()
    args.use_amp = not args.no_amp
    return args


def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(checkpoint_dir / "e2e_finetune.log")

    trainer = EndToEndTrainer(args, logger)
    trainer.fit(train_manifest=Path(args.train_manifest), val_manifest=Path(args.val_manifest))


if __name__ == "__main__":
    main()