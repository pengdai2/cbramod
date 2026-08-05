import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
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
    
    # Verify disk space
    stat = shutil.disk_usage(target_mount_dir)
    free_gb = stat.free / (1024**3)
    print(f"Target local block mount: {target_mount_dir} (Free Space: {free_gb:.2f} GB)")
    
    return target_mount_dir


def download_single_s3_file(
    endpoint_url: str,
    bucket_name: str,
    s3_key: str,
    local_file_path: Path,
    expected_size: int,
    pbar: tqdm,
    pbar_lock: Lock,
    force: bool = False
) -> bool:
    """
    Worker function to download a single file from S3 to local block storage.
    Skips downloading if the file already exists locally with matching file size.
    """
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable skip check: Local file exists and size matches S3
    if not force and local_file_path.exists() and local_file_path.stat().st_size == expected_size:
        with pbar_lock:
            pbar.update(expected_size)
        return True

    # Configure the client to skip optional checksums that break GCS
    gcs_config = Config(request_checksum_calculation="when_required")

    s3_client = boto3.client("s3", endpoint_url=endpoint_url, config=gcs_config)

    try:
        s3_client.download_file(
            Bucket=bucket_name,
            Key=s3_key,
            Filename=str(local_file_path)
        )
        with pbar_lock:
            pbar.update(expected_size)
        return True
    except Exception as e:
        print(f"\nFailed to download {s3_key}: {e}")
        return False


def sync_directory_from_s3(
    bucket_name: str,
    s3_prefix: str,
    target_dir: Path,
    endpoint_url: str = None,
    concurrency: int = 16,
    force: bool = False
) -> Path:
    """
    Recursively lists all objects under s3_prefix and downloads them in parallel
    directly to local NVMe/SSD block storage.
    """
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    s3_prefix = s3_prefix.strip("/")

    print(f"Listing S3 objects under s3://{bucket_name}/{s3_prefix}...")

    # 1. Paginate and list all S3 objects
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    s3_objects = []
    total_bytes = 0

    for page in pages:
        for obj in page.get("Contents", []):
            # Skip directory markers
            if obj["Key"].endswith("/"):
                continue
            s3_objects.append(obj)
            total_bytes += obj["Size"]

    if not s3_objects:
        raise FileNotFoundError(f"No objects found in s3://{bucket_name}/{s3_prefix}")

    print(f"Found {len(s3_objects):,} files totaling {total_bytes / (1024**3):.2f} GB.")
    print(f"Syncing to local storage ({target_dir}) using {concurrency} worker threads...\n")

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading Dataset")
    pbar_lock = Lock()

    # 2. Parallel multithreaded download
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for obj in s3_objects:
            key = obj["Key"]
            size = obj["Size"]

            # Compute relative path locally
            if s3_prefix and key.startswith(s3_prefix):
                rel_path = key[len(s3_prefix):].lstrip("/")
            else:
                rel_path = key

            local_file_path = target_dir / rel_path

            futures.append(
                executor.submit(
                    download_single_s3_file,
                    endpoint_url=endpoint_url,
                    bucket_name=bucket_name,
                    s3_key=key,
                    local_file_path=local_file_path,
                    expected_size=size,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    force=force
                )
            )

        results = [f.result() for f in as_completed(futures)]

    pbar.close()
    successful = sum(1 for r in results if r)

    if successful < len(s3_objects):
        raise RuntimeError(f"Download incomplete: {successful}/{len(s3_objects)} files downloaded successfully.")

    # Write marker file
    marker_file = target_dir / ".sync_complete"
    marker_file.touch()

    print(f"\n[SUCCESS] Synced {successful}/{len(s3_objects)} files to local storage.")
    return target_dir


def setup_local_block_storage(
    bucket_name: str,
    s3_prefix: str,
    mount_point: Path,
    endpoint_url: str = None,
    concurrency: int = 16,
    force_reprovision: bool = False
) -> Path:
    """End-to-end orchestration: Prepares block storage and streams S3 files in parallel."""
    mount_point = Path(mount_point).resolve()
    get_block_device_path(mount_point)

    dataset_target_path = mount_point / "dataset"
    dataset_target_path.mkdir(parents=True, exist_ok=True)

    marker_file = dataset_target_path / ".sync_complete"
    if marker_file.exists() and not force_reprovision:
        print(f"Dataset already fully synced to {dataset_target_path}. Skipping download.")
        print("Use --force to re-verify or force download.")
        return dataset_target_path

    # Download folder tree in parallel directly from S3
    sync_directory_from_s3(
        bucket_name=bucket_name,
        s3_prefix=s3_prefix,
        target_dir=dataset_target_path,
        endpoint_url=endpoint_url,
        concurrency=concurrency,
        force=force_reprovision
    )

    print(f"\n=== Local Storage Provisioning Complete ===")
    print(f"Dataset ready on NVMe/SSD at: {dataset_target_path}")
    return dataset_target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision Cloud Block Device by Syncing Uncompressed Directory from S3")
    parser.add_argument("--bucket", type=str, required=True, help="Cloud Object Storage Bucket")
    parser.add_argument("--s3_prefix", type=str, required=True, help="Remote S3 prefix folder path")
    parser.add_argument("--mount_point", type=str, default="/mnt/nvme/cbra_data", help="Local block storage path/mount point")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Custom S3 API endpoint URL (optional)")
    parser.add_argument("--concurrency", type=int, default=16, help="Max parallel download threads [Default: 16]")
    parser.add_argument("--force", action="store_true", help="Force re-verification/download of all files")

    args = parser.parse_args()

    setup_local_block_storage(
        bucket_name=args.bucket,
        s3_prefix=args.s3_prefix,
        mount_point=Path(args.mount_point),
        endpoint_url=args.endpoint_url,
        concurrency=args.concurrency,
        force_reprovision=args.force
    )
