import argparse
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from cbramod_common import CBraModE2EClassifier


def compute_multi_granularity_attributions(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    patch_size_samples: int = 200,  # CBraMod default 1-second patch at 200 Hz
    n_steps: int = 50,
    device: torch.device = torch.device("cuda")
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Integrated Gradients across three structural granularities:
    1. Signal level: [Channels, Time_Samples] (Full 200 Hz resolution)
    2. Patch level:  [Channels, Num_Patches]  (Aggregated across temporal patch windows)
    3. Channel level:[Channels]              (Spatial magnitude)
    """
    if not HAS_CAPTUM:
        raise ImportError("Captum library required. Run `pip install captum`.")

    model.eval()
    ig = IntegratedGradients(model)

    input_tensor = input_tensor.to(device)
    baseline = torch.zeros_like(input_tensor).to(device)

    # 1. Compute Raw Signal-Level Attributions [1, C, T]
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    signal_attr = attributions.squeeze(0).cpu().detach().numpy() # [Channels, Time_Samples]
    num_channels, time_samples = signal_attr.shape

    # 2. Compute Patch-Level Attributions [Channels, Num_Patches]
    # Reshape time dimension into discrete temporal patches
    num_patches = time_samples // patch_size_samples
    truncated_samples = num_patches * patch_size_samples
    
    # Reshape signal attributions to [Channels, Num_Patches, Patch_Size] and compute L2 norm per patch
    reshaped_attr = signal_attr[:, :truncated_samples].reshape(num_channels, num_patches, patch_size_samples)
    patch_attr = np.linalg.norm(reshaped_attr, axis=2) # [Channels, Num_Patches]

    # 3. Compute Spatial Channel-Level Attributions [Channels]
    channel_attr = np.linalg.norm(signal_attr, axis=1) # [Channels]

    return signal_attr, patch_attr, channel_attr


def plot_attribution_dashboard(
    raw_eeg: np.ndarray,
    signal_attr: np.ndarray,
    patch_attr: np.ndarray,
    channel_attr: np.ndarray,
    output_path: Path,
    target_channel_idx: int = 0,
    sfreq: float = 200.0
):
    """
    Generates a visual diagnostic dashboard comparing raw EEG signals, 
    signal-level saliency, and CBraMod patch grid importance.
    """
    time_axis = np.arange(raw_eeg.shape[1]) / sfreq
    num_channels, num_patches = patch_attr.shape

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1, 1.2]})

    # --- Panel 1: Raw EEG Trace with Overlay Signal Attribution ---
    eeg_signal = raw_eeg[target_channel_idx]
    attr_signal = signal_attr[target_channel_idx]
    
    axes[0].plot(time_axis, eeg_signal, color='black', alpha=0.6, label=f'Raw EEG (Ch {target_channel_idx})')
    # Overlay positive attribution as red highlights
    pos_attr = np.maximum(0, attr_signal)
    if pos_attr.max() > 0:
        pos_attr_norm = pos_attr / pos_attr.max() * np.abs(eeg_signal).max()
        axes[0].fill_between(time_axis, 0, pos_attr_norm, color='red', alpha=0.4, label='Positive Attribution')
        
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title(f'Signal-Level Attribution (Channel {target_channel_idx})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # --- Panel 2: CBraMod Spatial-Temporal Patch Heatmap ---
    im = axes[1].imshow(
        patch_attr, 
        aspect='auto', 
        cmap='magma', 
        origin='lower',
        extent=[0, time_axis[-1], 0, num_channels]
    )
    axes[1].set_ylabel('EEG Channel Index')
    axes[1].set_title('CBraMod Token Patch Importance Heatmap (Channels x Time Patches)')
    fig.colorbar(im, ax=axes[1], orientation='vertical', label='Patch Importance (L2 Norm)')

    # --- Panel 3: Global Channel Importance ---
    axes[2].bar(range(num_channels), channel_attr, color='navy', alpha=0.7)
    axes[2].set_xlabel('EEG Channel Index')
    axes[2].set_ylabel('Attribution L2 Norm')
    axes[2].set_title('Global Channel Importance Profile')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-granularity attribution plot saved to: {output_path}")


def analyze_sample_attributions(
    checkpoint_path: Path,
    sample_npy_path: Path,
    output_dir: Path,
    window_idx: int = 0,
    target_class: int = 1,
    num_channels: int = 64,
    device_str: str = "cuda"
):
    """Executes multi-granularity attribution on real preprocessed tensors."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    model = CBraModE2EClassifier(num_channels=num_channels, num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Load Sample
    raw_data = np.load(sample_npy_path)
    sample_slice = raw_data[window_idx : window_idx + 1]  # [1, 64, 6000]
    input_tensor = torch.from_numpy(sample_slice).float()

    # 3. Compute Attributions across Granularities
    signal_attr, patch_attr, channel_attr = compute_multi_granularity_attributions(
        model=model,
        input_tensor=input_tensor,
        target_class=target_class,
        patch_size_samples=200,  # 1s patch size
        device=device
    )

    # 4. Save Multi-Granularity Numpy Archives
    save_path = output_dir / f"multi_granularity_attr_w{window_idx}.npz"
    np.savez_compressed(
        save_path,
        signal_attribution=signal_attr, # [64, 6000]
        patch_attribution=patch_attr,   # [64, 30]
        channel_attribution=channel_attr,# [64]
        raw_eeg=sample_slice.squeeze(0)
    )
    print(f"Attribution dataset exported to {save_path}")

    # 5. Plot Diagnostic Dashboard
    plot_path = output_dir / f"attribution_dashboard_w{window_idx}.png"
    plot_attribution_dashboard(
        raw_eeg=sample_slice.squeeze(0),
        signal_attr=signal_attr,
        patch_attr=patch_attr,
        channel_attr=channel_attr,
        output_path=plot_path,
        target_channel_idx=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Granularity EEG Attribution Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--sample_npy", type=str, required=True, help="Path to sample .npy tensor")
    parser.add_argument("--output_dir", type=str, default="./attribution_results")
    parser.add_argument("--window_idx", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--num_channels", type=int, default=64)

    args = parser.parse_args()

    analyze_sample_attributions(
        checkpoint_path=Path(args.checkpoint),
        sample_npy_path=Path(args.sample_npy),
        output_dir=Path(args.output_dir),
        window_idx=args.window_idx,
        target_class=args.target_class,
        num_channels=args.num_channels
    )
