import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def check_prerequisites():
    """Verifies necessary environment variables and CLI tools are present."""
    required_env = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [var for var in required_env if not os.environ.get(var)]
    if missing:
        print(f"Warning: Missing environment variables: {', '.join(missing)}")
        print("Ensure cloud credentials are set before executing provisioning.")


def generate_cloud_init_script(
    bucket_name: str,
    s3_key: str,
    mount_point: str,
    git_repo_url: str
) -> str:
    """
    Generates a bash bootstrap script executed on instance boot (user data).
    Installs NVIDIA drivers, PyTorch, clones the repository, and mounts dataset.
    """
    bootstrap_bash = f"""#!/bin/bash
set -e

# Log execution output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting Compute Instance Bootstrap ==="

# 1. Update OS packages and install core tools
apt-get update -y
apt-get install -y git python3-pip python3-venv nvme-cli htop tar curl

# 2. Setup dedicated attached EBS data volume
if [ -b "/dev/nvme1n1" ]; then
    DATA_DEV="/dev/nvme1n1"
elif [ -b "/dev/sdb" ]; then
    DATA_DEV="/dev/sdb"
fi

if [ -n "$DATA_DEV" ]; then
    echo "Formatting attached block device $DATA_DEV as ext4..."
    mkfs.ext4 -F $DATA_DEV
    mkdir -p {mount_point}
    mount $DATA_DEV {mount_point}
    echo "$DATA_DEV {mount_point} ext4 defaults,nofail 0 2" >> /etc/fstab
fi

# 3. Create isolated virtual environment
python3 -m venv /opt/cbra_venv
source /opt/cbra_venv/bin/activate

# 4. Install PyTorch with CUDA 12 support and required EEG libraries
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mne yasa boto3 tqdm pandas scikit-learn matplotlib seaborn

# 5. Clone codebase repository
if [ -n "{git_repo_url}" ]; then
    echo "Cloning repository: {git_repo_url}..."
    git clone {git_repo_url} /opt/cbra_project
    cd /opt/cbra_project
fi

# 6. Run dataset storage provisioning from Cloud Object Storage
echo "Provisioning block storage dataset..."
python3 05_provision_block_storage.py \
    --bucket {bucket_name} \
    --s3_key {s3_key} \
    --mount_point {mount_point}

echo "=== Compute Instance Bootstrap Complete! ==="
"""
    return bootstrap_bash


def deploy_aws_ec2_instance(
    instance_type: str,
    ami_id: str,
    key_name: str,
    security_group: str,
    user_data_script: str,
    region: str = "us-east-1"
):
    """Provisions an AWS EC2 GPU instance using boto3."""
    import boto3

    ec2 = boto3.client("ec2", region_name=region)

    print(f"Launching AWS EC2 GPU instance ({instance_type})...")

    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group],
        MinCount=1,
        MaxCount=1,
        UserData=user_data_script,
        BlockDeviceMappings=[
            {
                # 1. OS Root Volume
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 100,  # OS & Environment
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True
                }
            },
            {
                # 2. Dedicated Secondary Data Volume for CBraMod Dataset
                "DeviceName": "/dev/sdb",
                "Ebs": {
                    "VolumeSize": 750,  # Fits full ~600GB uncompressed macro dataset
                    "VolumeType": "gp3",
                    "Iops": 6000,       # High throughput for fast PyTorch batch reads
                    "Throughput": 500,  # 500 MB/s
                    "DeleteOnTermination": True
                }
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "CBraMod-Training-Worker"}]
            }
        ]
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Instance launched successfully! ID: {instance_id}")
    print("Waiting for instance to reach 'running' state...")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Fetch public IP address
    info = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = info["Reservations"][0]["Instances"][0].get("PublicIpAddress")

    print(f"\n=== GPU Instance Provisioned ===")
    print(f"Instance ID: {instance_id}")
    print(f"Public IP:   {public_ip}")
    print(f"SSH Command: ssh -i ~/.ssh/{key_name}.pem ubuntu@{public_ip}")
    print("Bootstrap script is running in the background. Monitor logs via:")
    print(f"  ssh -i ~/.ssh/{key_name}.pem ubuntu@{public_ip} 'tail -f /var/log/user-data.log'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Compute Instance Provisioning & Bootstrap Script")
    parser.add_argument("--provider", type=str, choices=["aws"], default="aws", help="Cloud Provider")
    parser.add_argument("--instance_type", type=str, default="g5.2xlarge", help="GPU Instance Type (e.g., g5.2xlarge, p4d.24xlarge)")
    parser.add_argument("--ami_id", type=str, default="ami-0c7217cdde317cfec", help="Deep Learning AMI ID (Ubuntu 22.04 CUDA)")
    parser.add_argument("--key_name", type=str, required=True, help="SSH Key Pair Name")
    parser.add_argument("--security_group", type=str, required=True, help="Security Group ID (allowing SSH)")
    parser.add_argument("--bucket", type=str, required=True, help="Cloud Object Store Bucket")
    parser.add_argument("--s3_key", type=str, required=True, help="Dataset S3 archive key")
    parser.add_argument("--mount_point", type=str, default="/mnt/nvme/cbra_data", help="Local NVMe mount point")
    parser.add_argument("--repo_url", type=str, default="", help="Git repository URL to clone")

    args = parser.parse_args()

    check_prerequisites()

    bootstrap_script = generate_cloud_init_script(
        bucket_name=args.bucket,
        s3_key=args.s3_key,
        mount_point=args.mount_point,
        git_repo_url=args.repo_url
    )

    if args.provider == "aws":
        deploy_aws_ec2_instance(
            instance_type=args.instance_type,
            ami_id=args.ami_id,
            key_name=args.key_name,
            security_group=args.security_group,
            user_data_script=bootstrap_script
        )
