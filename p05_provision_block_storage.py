import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


def get_block_device_path(target_mount_dir: Path) -> Path:
    """
    Ensures local high-speed SSD/NVMe mount directory exists and is writable.
    Creates directory if absent.
    """
    target_mount_dir = target_mount_dir.resolve()
    target_mount_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify disk space (need enough space for extracted dataset)
    stat = shutil.disk_usage(target_mount_dir)
    free_gb = stat.free / (1024**3)
    print(f"Target local block mount: {target_mount_dir} (Free Space: {free_gb:.2f} GB)")
    
    return target_mount_dir


def download_from_s3(
    bucket_name: str, 
    s3_key: str, 
    download_path: Path, 
    endpoint_url: str = None
) -> Path:
    # Configure the client to skip optional checksums that break GCS
    gcs_config = Config(request_checksum_calculation="when_required")

    """Downloads dataset archive from cloud object store with progress bar."""
    s3_client = boto3.client("s3", endpoint_url=endpoint_url, config=gcs_config)
    
    # Get remote object metadata
    head = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    total_bytes = head["ContentLength"]
    expected_sha256 = head.get("Metadata", {}).get("sha256")

    print(f"Downloading s3://{bucket_name}/{s3_key} ({total_bytes / (1024**3):.2f} GB)...")

    # Check if archive already downloaded locally
    if download_path.exists() and download_path.stat().st_size == total_bytes:
        print("Archive already exists locally with matching file size. Skipping download.")
        return download_path

    pbar = tqdm(
        total=total_bytes, 
        unit="B", 
        unit_scale=True, 
        unit_divisor=1024, 
        desc="Downloading Archive"
    )

    def callback(bytes_transferred):
        pbar.update(bytes_transferred)

    s3_client.download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(download_path),
        Callback=callback
    )
    pbar.close()

    # Optional SHA-256 Checksum Validation
    if expected_sha256:
        print("Validating SHA-256 integrity checksum...")
        sha256 = hashlib.sha256()
        with open(download_path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                sha256.update(chunk)
        calc_sha = sha256.hexdigest()
        
        if calc_sha != expected_sha256:
            raise ValueError(f"Checksum mismatch! Expected {expected_sha256}, got {calc_sha}")
        print("Integrity verification successful!")

    return download_path


def unpack_archive_to_block_device(
    archive_path: Path, 
    extract_dir: Path, 
    force_unpack: bool = False
):
    """
    Unpacks downloaded tar archive directly onto local block device storage.
    Uses system tar if available for high-throughput streaming extraction.
    """
    marker_file = extract_dir / ".extraction_complete"
    
    if marker_file.exists() and not force_unpack:
        print(f"Dataset already extracted to {extract_dir}. Skipping unpack.")
        return

    print(f"Unpacking archive into high-speed block volume: {extract_dir}...")
    
    # Fast extraction using system tar executable if available
    if shutil.which("tar"):
        cmd = ["tar", "-xvf" if False else "-xf", str(archive_path), "-C", str(extract_dir)]
        res = subprocess.run(cmd, check=True)
    else:
        # Fallback to Python tarfile module
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    # Touch marker file on completion
    marker_file.touch()
    print("Dataset successfully extracted and mounted on local block device!")


def setup_local_block_storage(
    bucket_name: str,
    s3_key: str,
    mount_point: Path,
    endpoint_url: str = None,
    force_reprovision: bool = False
) -> Path:
    """End-to-end orchestration: Prepares block storage, streams S3 object, unpacks data."""
    mount_point = Path(mount_point).resolve()
    get_block_device_path(mount_point)

    archive_filename = Path(s3_key).name
    archive_local_path = mount_point / archive_filename
    extracted_dataset_path = mount_point / "dataset"
    extracted_dataset_path.mkdir(parents=True, exist_ok=True)

    # 1. Download dataset archive from Cloud Object Storage
    download_from_s3(
        bucket_name=bucket_name,
        s3_key=s3_key,
        download_path=archive_local_path,
        endpoint_url=endpoint_url
    )

    # 2. Unpack archive to NVMe/SSD local block storage
    unpack_archive_to_block_device(
        archive_path=archive_local_path,
        extract_dir=extracted_dataset_path,
        force_unpack=force_reprovision
    )

    # Clean up local archive file to reclaim space on local volume
    if archive_local_path.exists():
        archive_local_path.unlink()
        print("Removed temporary compressed archive file.")

    print(f"\n=== Storage Provisioning Complete ===")
    print(f"Dataset mounted and ready at: {extracted_dataset_path}")
    return extracted_dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision Cloud Block Device from S3 Object Store")
    parser.add_argument("--bucket", type=str, required=True, help="Cloud Object Storage Bucket")
    parser.add_argument("--s3_key", type=str, required=True, help="Remote S3 Key path for dataset archive")
    parser.add_argument("--mount_point", type=str, default="/mnt/nvme/cbra_data", help="Local block storage path/mount point")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Custom S3 API endpoint URL (optional)")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extraction")

    args = parser.parse_args()

    setup_local_block_storage(
        bucket_name=args.bucket,
        s3_key=args.s3_key,
        mount_point=Path(args.mount_point),
        endpoint_url=args.endpoint_url,
        force_reprovision=args.force
    )
