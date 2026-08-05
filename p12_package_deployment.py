import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
import torch

# Import architecture definition from our benchmark script
from cbramod_common import CBraModE2EClassifier


def create_deployment_package(
    checkpoint_path: Path,
    output_archive: Path,
    num_channels: int = 64,
    num_classes: int = 2,
    sample_rate: int = 200,
    window_sec: int = 30,
    patch_sec: int = 1,
    export_version: str = "1.0.0"
):
    """
    Bundles model weights, architecture metadata, and preprocessing specs 
    into a single compressed deployment archive (.tar.gz).
    """
    checkpoint_path = Path(checkpoint_path)
    output_archive = Path(output_archive)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    output_archive.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Packaging CBraMod Deployment Archive v{export_version} ===")

    # 1. Construct Configuration Metadata
    config_metadata = {
        "model_architecture": "CBraModRealWorldBenchmark",
        "version": export_version,
        "input_specs": {
            "num_channels": num_channels,
            "sample_rate_hz": sample_rate,
            "window_duration_sec": window_sec,
            "patch_duration_sec": patch_sec,
            "expected_shape": [num_channels, sample_rate * window_sec]
        },
        "output_specs": {
            "num_classes": num_classes,
            "classes": ["Control / Normal", "Abnormal / Target"] if num_classes == 2 else ["Wake", "N1", "N2", "N3", "REM"]
        },
        "pooling_strategy": "top_10_percentile"
    }

    # 2. Temporary Stage Directory for Assembly
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir)
        
        # Save Metadata JSON
        config_file = staging_path / "model_config.json"
        with open(config_file, "w") as f:
            json.dump(config_metadata, f, indent=2)
        print(f"  [+] Saved metadata config to: {config_file.name}")

        # Copy Weights File
        weights_file = staging_path / "model_weights.pt"
        shutil.copy(checkpoint_path, weights_file)
        print(f"  [+] Copied model checkpoint to: {weights_file.name}")

        # 3. Sanity Verification (Dry-Run Loading)
        print("  [*] Running pre-export sanity check...")
        try:
            model = CBraModE2EClassifier(num_channels=num_channels, num_classes=num_classes)
            checkpoint = torch.load(weights_file, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

            # Dummy input tensor: [1, Channels, Time_Samples]
            seq_len = sample_rate * window_sec
            dummy_input = torch.randn(1, num_channels, seq_len)
            with torch.no_grad():
                out = model(dummy_input)
            
            assert out.shape == (1, num_classes), f"Output shape mismatch: expected (1, {num_classes}), got {out.shape}"
            print("  [✓] Sanity check PASSED: Model instantiated and validated successfully.")

        except Exception as e:
            raise RuntimeError(f"Pre-export verification failed: {e}")

        # 4. Create Compressed Tarball Archive
        print(f"  [*] Compressing files into archive: {output_archive.name}...")
        with tarfile.open(output_archive, "w:gz") as tar:
            tar.add(config_file, arcname="model_config.json")
            tar.add(weights_file, arcname="model_weights.pt")

    print(f"\n[SUCCESS] Deployment archive successfully built at: {output_archive.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package CBraMod Weights & Pipeline into Deployment Archive")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint .pt file")
    parser.add_argument("--output_archive", type=str, default="./deploy/cbramod_deployment.tar.gz", help="Output .tar.gz archive path")
    parser.add_argument("--num_channels", type=int, default=64, help="Number of EEG channels")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of prediction classes")
    parser.add_argument("--sample_rate", type=int, default=200, help="EEG Sampling frequency (Hz)")

    args = parser.parse_args()

    create_deployment_package(
        checkpoint_path=Path(args.checkpoint),
        output_archive=Path(args.output_archive),
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        sample_rate=args.sample_rate
    )
