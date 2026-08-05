import argparse
import hashlib
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


def calculate_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculates SHA-256 checksum of local archive for data integrity validation."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def compress_directory(src_dir: Path, archive_path: Path) -> Path:
    """Compresses directory into a tar archive with progress bar."""
    print(f"Compressing '{src_dir}' into '{archive_path.name}'...")
    
    files = list(src_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    with tarfile.open(archive_path, "w:gz") as tar:
        pbar = tqdm(files, desc="Archiving Files", unit="file")
        for file in pbar:
            arcname = file.relative_to(src_dir.parent)
            tar.add(file, arcname=arcname)
            pbar.set_postfix({"File": file.name[:15]})

    print(f"Archive created. Size: {archive_path.stat().st_size / (1024**3):.2f} GB")
    return archive_path


def get_or_create_multipart_upload(s3_client, bucket: str, key: str, metadata: dict) -> str:
    """Finds an existing in-progress upload for key or initializes a new one."""
    paginator = s3_client.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for upload in page.get("Uploads", []):
            if upload["Key"] == key:
                print(f"Found active in-progress upload session: {upload['UploadId']}")
                return upload["UploadId"]

    response = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        Metadata=metadata
    )
    print(f"Created new multipart upload session: {response['UploadId']}")
    return response["UploadId"]


def get_completed_parts(s3_client, bucket: str, key: str, upload_id: str) -> dict:
    """Retrieves all already uploaded parts and their ETags."""
    completed_parts = {}
    paginator = s3_client.get_paginator("list_parts")
    try:
        for page in paginator.paginate(Bucket=bucket, Key=key, UploadId=upload_id):
            for part in page.get("Parts", []):
                completed_parts[part["PartNumber"]] = part["ETag"]
    except ClientError:
        pass
    return completed_parts


def upload_single_part(
    endpoint_url: str,
    bucket: str,
    key: str,
    upload_id: str,
    part_num: int,
    file_path: Path,
    offset: int,
    chunk_size: int,
    pbar: tqdm,
    pbar_lock: Lock
) -> dict:
    """Worker function to upload a single chunk from disk in parallel."""
    # Configure the client to skip optional checksums that break GCS
    gcs_config = Config(
        request_checksum_calculation="when_required"
    )

    # Create thread-local S3 client for thread safety
    s3_client = boto3.client("s3", endpoint_url=endpoint_url, config=gcs_config)

    with open(file_path, "rb") as f:
        f.seek(offset)
        chunk_data = f.read(chunk_size)

    response = s3_client.upload_part(
        Bucket=bucket,
        Key=key,
        PartNumber=part_num,
        UploadId=upload_id,
        Body=chunk_data
    )

    with pbar_lock:
        pbar.update(len(chunk_data))

    return {"PartNumber": part_num, "ETag": response["ETag"]}


def upload_to_object_store_resumable_parallel(
    local_file: Path,
    bucket_name: str,
    s3_key: str,
    endpoint_url: str = None,
    part_size_mb: int = 32,
    concurrency: int = 8
) -> bool:
    """
    Uploads local file to S3 with resumable multipart capability and multithreaded parallelism.
    """
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    file_size = local_file.stat().st_size
    chunk_size = part_size_mb * 1024 * 1024

    print("Calculating SHA-256 integrity checksum...")
    sha256_hash = calculate_sha256(local_file)
    print(f"SHA-256: {sha256_hash}")

    # 1. Skip if object exists and SHA matches
    try:
        head = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        if head.get("Metadata", {}).get("sha256") == sha256_hash:
            print(f"Remote object '{s3_key}' already exists with matching checksum. Skipping upload.")
            return True
    except ClientError:
        pass

    metadata = {
        "sha256": sha256_hash,
        "original_filename": local_file.name
    }

    # 2. Get active upload ID or start new session
    upload_id = get_or_create_multipart_upload(s3_client, bucket_name, s3_key, metadata)

    # 3. Retrieve parts that were already uploaded
    completed_parts = get_completed_parts(s3_client, bucket_name, s3_key, upload_id)
    if completed_parts:
        print(f"Resuming upload: {len(completed_parts)} part(s) already uploaded.")

    total_parts = (file_size + chunk_size - 1) // chunk_size
    parts_list = []

    pbar = tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Uploading {local_file.name}")
    pbar_lock = Lock()

    # Pre-fill ETags and progress bar for previously uploaded chunks
    missing_parts = []
    for part_num in range(1, total_parts + 1):
        if part_num in completed_parts:
            parts_list.append({"PartNumber": part_num, "ETag": completed_parts[part_num]})
            skipped_size = min(chunk_size, file_size - (part_num - 1) * chunk_size)
            pbar.update(skipped_size)
        else:
            missing_parts.append(part_num)

    # 4. Upload missing chunks in parallel
    if missing_parts:
        print(f"Uploading {len(missing_parts)} remaining part(s) using {concurrency} threads...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for part_num in missing_parts:
                offset = (part_num - 1) * chunk_size
                current_chunk_size = min(chunk_size, file_size - offset)
                
                futures.append(
                    executor.submit(
                        upload_single_part,
                        endpoint_url=endpoint_url,
                        bucket=bucket_name,
                        key=s3_key,
                        upload_id=upload_id,
                        part_num=part_num,
                        file_path=local_file,
                        offset=offset,
                        chunk_size=current_chunk_size,
                        pbar=pbar,
                        pbar_lock=pbar_lock
                    )
                )

            for future in as_completed(futures):
                part_result = future.result()
                parts_list.append(part_result)

    pbar.close()

    # 5. Finalize Multipart Upload
    print("Finalizing resumable multipart upload on S3...")
    s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=s3_key,
        UploadId=upload_id,
        MultipartUpload={"Parts": sorted(parts_list, key=lambda x: x["PartNumber"])}
    )
    
    print(f"\nSuccessfully uploaded to s3://{bucket_name}/{s3_key}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Compression and Resumable Parallel S3 Upload")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed dataset folder to upload")
    parser.add_argument("--archive_out", type=str, required=True, help="Path for temporary or local compressed archive (.tar.gz)")
    parser.add_argument("--bucket", type=str, required=True, help="Target Cloud Object Store S3 bucket name")
    parser.add_argument("--s3_key", type=str, required=True, help="Remote S3 Key path")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Custom S3 endpoint URL")
    parser.add_argument("--part_size_mb", type=int, default=32, help="Chunk size in MB for resumable upload [Default: 32MB]")
    parser.add_argument("--concurrency", type=int, default=8, help="Max parallel upload threads [Default: 8]")
    parser.add_argument("--skip_compression", action="store_true", help="If specified, uploads an existing archive file directly")

    args = parser.parse_args()

    archive_path = Path(args.archive_out)
    data_dir = Path(args.data_dir)

    # 1. Compress directory if needed
    if not args.skip_compression or not archive_path.is_file():
        compress_directory(data_dir, archive_path)
    else:
        print(f"Skipping compression. Using existing archive at {archive_path}")

    # 2. Resumable Parallel Upload
    upload_to_object_store_resumable_parallel(
        local_file=archive_path,
        bucket_name=args.bucket,
        s3_key=args.s3_key,
        endpoint_url=args.endpoint_url,
        part_size_mb=args.part_size_mb,
        concurrency=args.concurrency
    )
