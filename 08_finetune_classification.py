import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

# Import model architecture setup from benchmark module
from real_world_benchmark import CBraModRealWorldBenchmark, RealSleepEEGDataset


class EarlyStopping:
    """Early stopping handler based on Macro F1 validation score."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1: float) -> bool:
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement in Val F1 for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0
        return self.early_stop


def setup_data_loader_and_criterion(
    train_ds: RealSleepEEGDataset,
    batch_size: int,
    num_workers: int,
    imbalance_strategy: str,
    device: torch.device
) -> Tuple[DataLoader, nn.Module]:
    """
    Configures DataLoader and CrossEntropy Loss function based on chosen imbalance strategy.
    
    Strategies:
      - 'sampler': Uses WeightedRandomSampler to oversample minority classes dynamically. Standard CrossEntropyLoss.
      - 'loss_weights': Standard DataLoader with shuffle=True. Weighted CrossEntropyLoss using inverse frequency weights.
      - 'none': Standard DataLoader with shuffle=True and unweighted CrossEntropyLoss.
    """
    train_labels = np.array([sample[2] for sample in train_ds.samples])
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)

    print("\n--- Training Set Class Distribution ---")
    for class_id, count in enumerate(class_counts):
        pct = (count / total_samples) * 100.0
        print(f"  Class {class_id}: {count:,} samples ({pct:.2f}%)")

    if imbalance_strategy == "sampler":
        print("--> Imbalance Strategy: WeightedRandomSampler (Resampling minority class batches)")
        class_weights_raw = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights_raw[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            pin_memory=True, persistent_workers=True, prefetch_factor=2
        )
        criterion = nn.CrossEntropyLoss()

    elif imbalance_strategy == "loss_weights":
        print("--> Imbalance Strategy: Class-Weighted CrossEntropyLoss")
        # Standard balanced weight formula: total / (K * N_c)
        balanced_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
        weights_tensor = torch.from_numpy(balanced_weights).float().to(device)
        
        weight_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(balanced_weights)])
        print(f"    Derived Loss Weights: [{weight_str}]")

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=True, prefetch_factor=2
        )

    elif imbalance_strategy == "none":
        print("--> Imbalance Strategy: None (Unweighted training)")
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=True, prefetch_factor=2
        )

    else:
        raise ValueError(f"Invalid imbalance strategy: {imbalance_strategy}. Choose 'sampler', 'loss_weights', or 'none'.")

    return train_loader, criterion


def run_fine_tuning_pipeline(
    train_manifest: Path,
    val_manifest: Path,
    data_dir: Optional[Path] = None,
    num_channels: int = 64,
    num_classes: int = 2,
    batch_size: int = 64,
    epochs: int = 20,
    lr_backbone: float = 1e-5,
    lr_head: float = 3e-4,
    weight_decay: float = 1e-4,
    imbalance_strategy: str = "sampler",
    num_workers: int = 4,
    checkpoint_dir: Path = Path("./checkpoints"),
    device_str: str = "cuda"
):
    """Executes fine-tuning pipeline with validation logging and checkpointing."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "cbramod_best_finetuned.pt"
    latest_model_path = checkpoint_dir / "cbramod_latest.pt"

    print(f"\n=== Initializing CBraMod Fine-Tuning Pipeline on [{device}] ===")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    if data_dir:
        print(f"Data Root Directory:  {data_dir}")

    # 1. Instantiate Datasets
    train_ds = RealSleepEEGDataset(train_manifest, data_dir=data_dir)
    val_ds = RealSleepEEGDataset(val_manifest, data_dir=data_dir)

    # 2. Configure Strategy-Based DataLoader & Loss Criterion
    train_loader, criterion = setup_data_loader_and_criterion(
        train_ds=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        imbalance_strategy=imbalance_strategy,
        device=device
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # 3. Instantiate Model Architecture
    model = CBraModRealWorldBenchmark(num_channels=num_channels, num_classes=num_classes).to(device)

    # 4. Configure Differential Learning Rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head}
    ], weight_decay=weight_decay)

    early_stopper = EarlyStopping(patience=5)

    start_epoch = 1
    best_val_f1 = 0.0

    # Auto-resume checkpoint logic
    if latest_model_path.exists():
        print(f"\n[AUTO-RESUME] Resuming from checkpoint: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", checkpoint.get("val_f1", 0.0))
        
        if "early_stop_counter" in checkpoint:
            early_stopper.counter = checkpoint["early_stop_counter"]
        if "early_stop_best_score" in checkpoint:
            early_stopper.best_score = checkpoint["early_stop_best_score"]

        print(f"[AUTO-RESUME] Resuming at Epoch {start_epoch} (Best Val F1: {best_val_f1:.4f})")

    print(f"Backbone LR: {lr_backbone} | Head LR: {lr_head} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 70)

    # 5. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = (train_correct / train_total * 100.0) if train_total > 0 else 0.0

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  ", leave=False)
        with torch.no_grad():
            for x_batch, y_batch in val_pbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

                val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_val_loss = val_loss / len(val_targets) if len(val_targets) > 0 else 0.0
        epoch_val_acc = accuracy_score(val_targets, val_preds) * 100.0 if len(val_targets) > 0 else 0.0
        epoch_val_f1 = f1_score(val_targets, val_preds, average="macro") if len(val_targets) > 0 else 0.0
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, Macro F1: {epoch_val_f1:.4f}"
        )

        # Save Best Model
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": epoch_val_acc
            }, best_model_path)
            print(f"  --> [SAVED BEST MODEL] Best Val F1: {best_val_f1:.4f} saved to {best_model_path}")

        # Save Latest Model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "val_f1": epoch_val_f1,
            "val_acc": epoch_val_acc,
            "early_stop_counter": early_stopper.counter,
            "early_stop_best_score": early_stopper.best_score
        }, latest_model_path)

        # Early Stopping Check
        if early_stopper(epoch_val_f1):
            print(f"\n[EARLY STOPPING TRIGGERED] Terminated at epoch {epoch}.")
            break

    print("\n" + "=" * 70)
    print(f"Fine-Tuning Complete. Best Validation Macro F1 Score: {best_val_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBraMod Fine-Tuning Pipeline with Real Preprocessed Data")
    parser.add_argument("--manifest_dir", type=str, required=True, help="Directory containing train_manifest.csv and val_manifest.csv")
    parser.add_argument("--data_dir", type=str, default=None, help="Top-level root directory where relative tensor/meta files reside")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory where checkpoints are saved/resumed")
    parser.add_argument("--epochs", type=int, default=20, help="Target training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num_channels", type=int, default=64, help="EEG Channel count")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of target classes")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--lr_head", type=float, default=3e-4, help="Classification head learning rate (default: 3e-4)")
    parser.add_argument(
        "--imbalance_strategy", 
        type=str, 
        choices=["sampler", "loss_weights", "none"], 
        default="sampler", 
        help="Class imbalance mitigation strategy: 'sampler' (WeightedRandomSampler), 'loss_weights' (Weighted CrossEntropyLoss), or 'none'"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader CPU workers")

    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    checkpoint_dir = Path(args.checkpoint_dir)
    train_csv = manifest_dir / "train_manifest.csv"
    val_csv = manifest_dir / "val_manifest.csv"

    run_fine_tuning_pipeline(
        train_manifest=train_csv,
        val_manifest=val_csv,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        imbalance_strategy=args.imbalance_strategy,
        num_workers=args.num_workers
    )
