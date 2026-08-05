import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np

from cbramod_common import fetch_and_preprocess_sleep_edf, CBraModRealWorldBenchmark

def verify_cbramod_checkpoints(subject_id: int = 0, epochs: int = 10, batch_size: int = 16, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"=== Starting CBraMod Pipeline Checkpoint Verification on [{device}] ===")

    # 1. Fetch & Preprocess Data
    eeg_data, target_labels, sfreq = fetch_and_preprocess_sleep_edf(subject_id=subject_id)
    
    x_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    y_tensor = torch.tensor(target_labels, dtype=torch.long)

    # 80/20 Train/Val Split on single subject epochs
    num_samples = len(x_tensor)
    train_size = int(0.8 * num_samples)
    
    train_x, val_x = x_tensor[:train_size], x_tensor[train_size:]
    train_y, val_y = y_tensor[:train_size], y_tensor[train_size:]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    # 2. Build Model & Setup Training
    num_channels = eeg_data.shape[1]
    model = CBraModRealWorldBenchmark(num_channels=num_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n" + "="*50)
    print(" CHECKPOINT A & B: Training Convergence Verification")
    print("="*50)

    stage_names = ["Wake", "N1", "N2", "N3", "REM"]

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Output Epoch Status
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # Checkpoint A Verification (Epoch 1)
        if epoch == 1:
            print("\n  --> [CHECKPOINT A EVALUATION]:")
            if 1.30 <= epoch_loss <= 1.65:
                print(f"      [PASS] Initial Loss ({epoch_loss:.4f}) is near theoretically expected ~1.609.")
            else:
                print(f"      [WARNING] Initial Loss ({epoch_loss:.4f}) deviates from expected range.")

    # Checkpoint B Verification (Epoch 10)
    print("\n  --> [CHECKPOINT B EVALUATION]:")
    if epoch_loss < 0.80:
        print(f"      [PASS] Model successfully converged. Final Loss: {epoch_loss:.4f}")
    else:
        print(f"      [WARNING] High final loss ({epoch_loss:.4f}). Optimization may be stalling.")

    # 3. Checkpoint C: Evaluate Validation Metrics & Per-Class F1
    print("\n" + "="*50)
    print(" CHECKPOINT C: Per-Class F1 & Literature Baseline Evaluation")
    print("="*50)

    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.numpy())

    val_acc = accuracy_score(val_targets, val_preds) * 100.0
    macro_f1 = f1_score(val_targets, val_preds, average="macro")

    print(f"\nValidation Accuracy: {val_acc:.2f}%")
    print(f"Validation Macro F1: {macro_f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(val_targets, val_preds, target_names=stage_names, zero_division=0))

if __name__ == "__main__":
    verify_cbramod_checkpoints(subject_id=0, epochs=10, batch_size=16)
