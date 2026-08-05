import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


def calculate_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculates SHA-256 checksum of a local file for data integrity validation."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_or_create_multipart_upload(s3_client, bucket: str, key: str, metadata: dict) -> str:
    """Finds an existing in-progress upload for a specific key or initializes a new one."""
    paginator = s3_client.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for upload in page.get("Uploads", []):
            if upload["Key"] == key:
                return upload["UploadId"]

    response = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        Metadata=metadata
    )
    return response["UploadId"]


def get_completed_parts(s3_client, bucket: str, key: str, upload_id: str) -> dict:
    """Retrieves already uploaded parts and ETags for a multipart session."""
    completed_parts = {}
    paginator = s3_client.get_paginator("list_parts")
    try:
        for page in paginator.paginate(Bucket=bucket, Key=key, UploadId=upload_id):
            for part in page.get("Parts", []):
                completed_parts[part["PartNumber"]] = part["ETag"]
    except ClientError:
        pass
    return completed_parts


def upload_single_file(
    endpoint_url: str,
    bucket_name: str,
    local_file: Path,
    s3_key: str,
    part_size_mb: int,
    pbar: tqdm,
    pbar_lock: Lock
) -> bool:
    """
    Worker function to handle uploading a single file (small or large multipart)
    with skip checks and error handling.
    """
    # Configure the client to skip optional checksums that break GCS
    gcs_config = Config(request_checksum_calculation="when_required")

    # Create thread-local S3 client
    s3_client = boto3.client("s3", endpoint_url=endpoint_url, config=gcs_config)
    file_size = local_file.stat().st_size
    chunk_size = part_size_mb * 1024 * 1024

    # 1. Skip check: File exists remotely with matching size
    try:
        head = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        if head.get("ContentLength") == file_size:
            with pbar_lock:
                pbar.update(file_size)
            return True
    except ClientError:
        pass  # File does not exist remotely, proceed with upload

    # 2. Small File Upload (< part_size_mb)
    if file_size <= chunk_size:
        try:
            with open(local_file, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=f.read()
                )
            with pbar_lock:
                pbar.update(file_size)
            return True
        except Exception as e:
            print(f"\nFailed to upload small file {local_file.name}: {e}")
            return False

    # 3. Large File Multipart Upload (Resumable)
    try:
        sha256_hash = calculate_sha256(local_file)
        metadata = {"sha256": sha256_hash, "original_filename": local_file.name}
        
        upload_id = get_or_create_multipart_upload(s3_client, bucket_name, s3_key, metadata)
        completed_parts = get_completed_parts(s3_client, bucket_name, s3_key, upload_id)

        total_parts = (file_size + chunk_size - 1) // chunk_size
        parts_list = []

        with open(local_file, "rb") as f:
            for part_num in range(1, total_parts + 1):
                offset = (part_num - 1) * chunk_size
                current_chunk_size = min(chunk_size, file_size - offset)

                if part_num in completed_parts:
                    parts_list.append({"PartNumber": part_num, "ETag": completed_parts[part_num]})
                    with pbar_lock:
                        pbar.update(current_chunk_size)
                    continue

                f.seek(offset)
                chunk_data = f.read(current_chunk_size)

                response = s3_client.upload_part(
                    Bucket=bucket_name,
                    Key=s3_key,
                    PartNumber=part_num,
                    UploadId=upload_id,
                    Body=chunk_data
                )
                parts_list.append({"PartNumber": part_num, "ETag": response["ETag"]})
                
                with pbar_lock:
                    pbar.update(len(chunk_data))

        s3_client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": sorted(parts_list, key=lambda x: x["PartNumber"])}
        )
        return True

    except Exception as e:
        print(f"\nFailed multipart upload for {local_file.name}: {e}")
        return False


def sync_directory_to_s3(
    data_dir: Path,
    bucket_name: str,
    s3_prefix: str,
    endpoint_url: str = None,
    part_size_mb: int = 32,
    concurrency: int = 16
):
    """
    Recursively scans local folder and uploads all files in parallel.
    Preserves directory structure relative to data_dir root.
    """
    data_dir = Path(data_dir).resolve()
    print(f"Scanning directory '{data_dir}' for files...")
    
    all_files = [f for f in data_dir.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in all_files)

    print(f"Found {len(all_files):,} files totaling {total_bytes / (1024**3):.2f} GB.")
    print(f"Uploading using {concurrency} worker threads...\n")

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="Syncing Directory")
    pbar_lock = Lock()

    s3_prefix = s3_prefix.strip("/")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for file_path in all_files:
            # Map local relative path to S3 object key
            rel_path = file_path.relative_to(data_dir)
            s3_key = f"{s3_prefix}/{rel_path}" if s3_prefix else str(rel_path)

            futures.append(
                executor.submit(
                    upload_single_file,
                    endpoint_url=endpoint_url,
                    bucket_name=bucket_name,
                    local_file=file_path,
                    s3_key=s3_key,
                    part_size_mb=part_size_mb,
                    pbar=pbar,
                    pbar_lock=pbar_lock
                )
            )

        # Wait for all files to finish
        results = [future.result() for future in as_completed(futures)]

    pbar.close()
    successful = sum(1 for r in results if r)
    print(f"\n[COMPLETE] Uploaded/Verified {successful}/{len(all_files)} files to s3://{bucket_name}/{s3_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Resumable Uncompressed Directory Sync to S3/GCS")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed dataset folder")
    parser.add_argument("--bucket", type=str, required=True, help="Target Object Store S3 bucket name")
    parser.add_argument("--s3_prefix", type=str, default="", help="Target directory prefix on S3 (e.g., 'datasets/sleep_cohort')")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Custom S3 endpoint (e.g. https://storage.googleapis.com)")
    parser.add_argument("--part_size_mb", type=int, default=32, help="Multipart chunk threshold in MB [Default: 32MB]")
    parser.add_argument("--concurrency", type=int, default=16, help="Max parallel file upload threads [Default: 16]")

    args = parser.parse_args()

    sync_directory_to_s3(
        data_dir=Path(args.data_dir),
        bucket_name=args.bucket,
        s3_prefix=args.s3_prefix,
        endpoint_url=args.endpoint_url,
        part_size_mb=args.part_size_mb,
        concurrency=args.concurrency
    )
