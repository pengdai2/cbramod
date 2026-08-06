import argparse
import time
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.datasets.sleep_physionet import age

from cbramod_common import CBraModE2EClassifier


def fetch_and_preprocess_sleep_edf(subject_id: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Downloads a real 8-hour sleep EEG recording from PhysioNet Sleep-EDF,
    resamples to 200 Hz, extracts 30-second epochs, and maps ground-truth sleep stages.
    """
    print(f"--> Fetching real-world Sleep-EDF data for Subject {subject_id} from PhysioNet...")
    [file_path] = age.fetch_data(subjects=[subject_id], recording=[1])

    # Load raw EDF recording and corresponding annotations (.edf / .txt)
    raw = mne.io.read_raw_edf(file_path[0], preload=True, verbose=False)
    annot = mne.read_annotations(file_path[1])
    raw.set_annotations(annot, emit_warning=False)

    # Standardize channel configuration and apply 0.5-35 Hz bandpass filter
    raw.filter(l_freq=0.5, h_freq=35.0, verbose=False)

    # Map AASM Sleep Stage annotations to integer target classes
    annotation_mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,  # Merge N4 into N3 per AASM guidelines
        "Sleep stage R": 4
    }

    # Extract 30-second continuous epochs matching AASM staging
    events, event_id = mne.events_from_annotations(
        raw, event_id=annotation_mapping, chunk_duration=30.0, verbose=False
    )

    tmax = 30.0 - (1.0 / raw.info["sfreq"])
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Resample epochs to CBraMod's target 200 Hz sampling rate
    print("--> Resampling real EEG epochs to 200 Hz...")
    epochs.resample(sfreq=200.0, verbose=False)

    data = epochs.get_data(units="uV")  # Shape: [Epochs, Channels, Time_Samples (6000)]
    labels = epochs.events[:, -1]       # Integer sleep stage labels (0..4)

    return data, labels, epochs.info["sfreq"]


def verify_cbramod_pipeline(subject_id: int = 0, epochs: int = 10, batch_size: int = 16, device_str: str = "cuda"):
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
    model = CBraModE2EClassifier(num_channels=num_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n" + "="*50)
    print(" CHECKPOINT A & B: Training Convergence Verification")
    print("="*50)

    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    stage_labels = {i: name for i, name in enumerate(stage_names)}  

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
    print(classification_report(val_targets, val_preds, labels=list(stage_labels.keys()), target_names=list(stage_labels.values()), zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBraMod Verification on PhysioNet Data")
    parser.add_argument("--subject_id", type=int, default=0, help="PhysioNet Subject ID to fetch")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference/Training Batch Size")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device (cuda/cpu)")

    args = parser.parse_args()

    verify_cbramod_pipeline(
        subject_id=args.subject_id,
        batch_size=args.batch_size,
        device_str=args.device)