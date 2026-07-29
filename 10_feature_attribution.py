import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import Captum for feature attribution
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

# Import model architecture setup from our benchmark module
from 07b_real_world_benchmark import CBraModRealWorldBenchmark


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda")
) -> np.ndarray:
    """
    Computes Integrated Gradients attribution maps for a given input window tensor.
    
    Args:
        model: Fine-tuned PyTorch model
        input_tensor: Shape [1, Channels, Time_Samples] e.g., [1, 64, 6000]
        target_class: Target class index to explain (0 or 1)
        n_steps: Number of interpolation steps for Riemann sum approximation
        
    Returns:
        attributions: Array of same shape as input [64, 6000] containing importance scores.
    """
    if not HAS_CAPTUM:
        raise ImportError("Captum library is required for feature attribution. Install via `pip install captum`.")

    model.eval()
    ig = IntegratedGradients(model)

    input_tensor = input_tensor.to(device)
    baseline = torch.zeros_like(input_tensor).to(device)

    # Compute attributions relative to zero baseline
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    print(f"  [Captum] IG Convergence Delta: {delta.item():.6f}")
    return attributions.squeeze(0).cpu().detach().numpy()


def analyze_channel_attribution(
    checkpoint_path: Path,
    sample_npy_path: Path,
    output_dir: Path,
    window_idx: int = 0,
    target_class: int = 1,
    num_channels: int = 64,
    device_str: str = "cuda"
):
    """Loads model checkpoint, executes Integrated Gradients, and exports saliency maps."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running Integrated Gradients Attribution Analysis on [{device}] ===")

    # 1. Instantiate Model & Load Weights
    model = CBraModRealWorldBenchmark(num_channels=num_channels, num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load Sample EEG Window Tensor
    raw_data = np.load(sample_npy_path)
    if window_idx >= len(raw_data):
        raise ValueError(f"Window index {window_idx} out of bounds for tensor with {len(raw_data)} slices.")

    window_sample = raw_data[window_idx : window_idx + 1]  # Shape: [1, 64, 6000]
    input_tensor = torch.from_numpy(window_sample).float()

    # 3. Compute Attribution
    attr_map = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_class=target_class,
        n_steps=50,
        device=device
    )  # Shape: [64, 6000]

    # 4. Aggregated Channel Importance (L2 Norm across time samples)
    channel_importance = np.linalg.norm(attr_map, axis=1)  # Shape: [64]
    
    # Save Raw Attribution Arrays
    save_path = output_dir / f"ig_attribution_w{window_idx}_c{target_class}.npz"
    np.savez_compressed(
        save_path,
        attributions=attr_map,
        channel_importance=channel_importance,
        raw_eeg=window_sample.squeeze(0)
    )
    print(f"Attribution arrays saved to: {save_path}")

    # 5. Export Summary Bar Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(num_channels), channel_importance, color="skyblue", edgecolor="navy")
    plt.xlabel("EEG Channel Index")
    plt.ylabel("Attribution Magnitude (L2 Norm)")
    plt.title(f"Channel Importance Profile (Target Class: {target_class})")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plot_path = output_dir / f"channel_importance_w{window_idx}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Channel importance plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Gradients EEG Feature Attribution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned checkpoint (.pt)")
    parser.add_argument("--sample_npy", type=str, required=True, help="Path to subject window .npy array")
    parser.add_argument("--output_dir", type=str, default="./attribution_results", help="Output directory")
    parser.add_argument("--window_idx", type=int, default=0, help="Index of window slice to analyze")
    parser.add_argument("--target_class", type=int, default=1, help="Target class index")
    parser.add_argument("--num_channels", type=int, default=64, help="Number of EEG channels")

    args = parser.parse_args()

    analyze_channel_attribution(
        checkpoint_path=Path(args.checkpoint),
        sample_npy_path=Path(args.sample_npy),
        output_dir=Path(args.output_dir),
        window_idx=args.window_idx,
        target_class=args.target_class,
        num_channels=args.num_channels
    )