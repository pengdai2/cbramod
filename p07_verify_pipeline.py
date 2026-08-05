import argparse
import time
from cbramod_common import SyntheticEEGDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from braindecode.models import CBraMod


def verify_pipeline_setup(
    batch_size: int = 16, 
    num_samples: int = 128, 
    device_str: str = "cuda"
):
    """Executes a forward/backward pass and measures throughput (samples/sec)."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"=== Running CBraMod Pipeline Verification on [{device}] ===")

    # 1. Instantiate Data Pipeline
    dataset = SyntheticEEGDataset(num_samples=num_samples, channels=64, time_samples=6000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Build CBraMod Model & Optimizer
    model = CBraMod(
        n_outputs=2,
        n_chans=64,
        sfreq=200.0,
        return_encoder_output=False
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    start_time = time.time()
    total_loss = 0.0

    print("Executing forward & backpropagation verification steps...")
    for step, (x_batch, y_batch) in enumerate(loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"  Step [{step+1}/{len(loader)}] - Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    throughput = num_samples / elapsed

    print("\n=== Pipeline Verification Summary ===")
    print(f"  - Total Elapsed Time: {elapsed:.2f} seconds")
    print(f"  - Processing Throughput: {throughput:.2f} samples/sec")
    print(f"  - Mean Batch Loss: {total_loss / len(loader):.4f}")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"  - Peak VRAM Allocation: {peak_mem:.2f} MB")

    print("\n[SUCCESS] CBraMod pipeline setup verified! Architecture and gradients functioning properly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBraMod Pipeline Verification")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the test")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of synthetic samples")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device (cuda/cpu)")

    args = parser.parse_args()

    verify_pipeline_setup(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device_str=args.device
    )