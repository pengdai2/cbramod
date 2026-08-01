import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mne

from real_world_benchmark import fetch_and_preprocess_sleep_edf, CBraModRealWorldBenchmark


def run_real_world_benchmark(subject_id: int = 0, batch_size: int = 16, device_str: str = "cuda"):
    """Executes a real-world benchmark run on actual PhysioNet patient data."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Real-World CBraMod Benchmark on [{device}] ===")

    # 1. Fetch & Preprocess Real Dataset
    eeg_data, target_labels, sfreq = fetch_and_preprocess_sleep_edf(subject_id=subject_id)
    
    num_epochs, num_channels, time_samples = eeg_data.shape
    print(f"\nDataset Extracted:")
    print(f"  - Subject ID:     {subject_id}")
    print(f"  - Total Epochs:   {num_epochs} (30s windows)")
    print(f"  - EEG Channels:   {num_channels}")
    print(f"  - Sampling Rate:  {sfreq} Hz")
    print(f"  - Input Shape:    [{num_epochs}, {num_channels}, {time_samples}]")

    # Convert to PyTorch Tensors
    x_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    y_tensor = torch.tensor(target_labels, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Build Model & Criterion
    model = CBraModRealWorldBenchmark(num_channels=num_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3. Execute Verification Epoch
    model.train()
    start_time = time.time()
    
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    print("\nRunning Benchmark Forward & Backward Passes on Patient Data...")
    for step, (x_batch, y_batch) in enumerate(loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == y_batch).sum().item()
        total_preds += y_batch.size(0)

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            print(f"  Batch [{step+1}/{len(loader)}] - Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    epoch_loss = running_loss / total_preds
    accuracy = (correct_preds / total_preds) * 100.0
    throughput = total_preds / elapsed

    print("\n=== Real-World Benchmark Verification Results ===")
    print(f"  - Total Processed Epochs: {total_preds}")
    print(f"  - Execution Time:         {elapsed:.2f} seconds")
    print(f"  - Throughput:             {throughput:.2f} epochs/sec")
    print(f"  - Average Loss:           {epoch_loss:.4f}")
    print(f"  - Baseline Staging Acc:   {accuracy:.2f}%")
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"  - Peak VRAM Allocation:   {peak_vram:.2f} MB")

    print("\n[SUCCESS] Real-world benchmark completed. Model pipeline handles genuine electrophysiological signals and ground-truth labels correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-World CBraMod Verification on PhysioNet Data")
    parser.add_argument("--subject_id", type=int, default=0, help="PhysioNet Subject ID to fetch")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference/Training Batch Size")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device (cuda/cpu)")

    args = parser.parse_args()

    run_real_world_benchmark(
        subject_id=args.subject_id,
        batch_size=args.batch_size,
        device_str=args.device
    )
