import argparse
import os
import sys
import time
from pathlib import Path


def check_prerequisites():
    """Verifies standard Google Cloud environment variable or project ID setup."""
    project_id = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("Warning: GCP_PROJECT environment variable is not explicitly set.")
        print("The script will fall back to default project configured in gcloud CLI.")


def generate_cloud_init_script(
    bucket_name: str,
    s3_prefix: str,
    mount_point: str,
    git_repo_url: str,
    endpoint_url: str = "https://storage.googleapis.com"
) -> str:
    """
    Generates bash bootstrap script executed on GCE VM boot via metadata startup-script.
    """
    bootstrap_bash = f"""#!/bin/bash
set -e

# Log execution output
exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

echo "=== Starting GCP Compute Instance Bootstrap ==="

# 1. Update OS packages and install core tools
apt-get update -y
apt-get install -y git python3-pip python3-venv nvme-cli htop tar curl

# 2. Setup attached secondary persistent block disk
DATA_DEV=""
if [ -b "/dev/sdb" ]; then
    DATA_DEV="/dev/sdb"
elif [ -b "/dev/disk/by-id/google-persistent-disk-1" ]; then
    DATA_DEV="/dev/disk/by-id/google-persistent-disk-1"
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

# 4. Install PyTorch with CUDA support and dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mne yasa boto3 tqdm pandas scikit-learn matplotlib seaborn google-cloud-storage

# 5. Clone codebase repository
if [ -n "{git_repo_url}" ]; then
    echo "Cloning repository: {git_repo_url}..."
    git clone {git_repo_url} /opt/cbra_project
    cd /opt/cbra_project
fi

# 6. Run uncompressed directory sync from GCS/S3 using parallel downloader
echo "Provisioning local storage from object store..."
python3 05_provision_local_storage.py \\
    --bucket {bucket_name} \\
    --s3_prefix {s3_prefix} \\
    --mount_point {mount_point} \\
    --endpoint_url {endpoint_url} \\
    --concurrency 32

echo "=== Compute Instance Bootstrap Complete! ==="
"""
    return bootstrap_bash


def deploy_gcp_compute_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    user_data_script: str,
    disk_size_gb: int = 750
):
    """Provisions a Google Cloud Compute Engine GPU Instance using Google Cloud SDK."""
    from google.cloud import compute_v1

    instance_client = compute_v1.InstancesClient()

    print(f"Launching GCP GPU Instance '{instance_name}' in zone {zone} ({machine_type})...")

    # 1. Configure OS Boot Disk (Deep Learning PyTorch Image)
    boot_disk = compute_v1.AttachedDisk()
    boot_disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        disk_size_gb=100,
        source_image="projects/ml-images/global/images/family/c0-deeplearning-common-cu121-ubuntu-2204"
    )
    boot_disk.auto_delete = True
    boot_disk.boot = True

    # 2. Configure High-Performance Secondary Block Disk for Dataset
    data_disk = compute_v1.AttachedDisk()
    data_disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        disk_size_gb=disk_size_gb,
        disk_type=f"zones/{zone}/diskTypes/pd-ssd",  # Fast SSD persistent disk
        auto_delete=True
    )
    data_disk.auto_delete = True
    data_disk.boot = False

    # 3. Configure GPU Accelerator (e.g. NVIDIA L4 or T4)
    guest_accelerators = []
    if accelerator_type:
        accelerator = compute_v1.AcceleratorConfig()
        accelerator.accelerator_type = f"zones/{zone}/acceleratorTypes/{accelerator_type}"
        accelerator.accelerator_count = accelerator_count
        guest_accelerators.append(accelerator)

    # 4. Network Interface with External Public IP
    network_interface = compute_v1.NetworkInterface()
    network_interface.name = "global/networks/default"
    access_config = compute_v1.AccessConfig()
    access_config.name = "External NAT"
    access_config.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
    network_interface.access_configs.append(access_config)

    # 5. Pass Startup Script in Metadata
    metadata = compute_v1.Metadata()
    metadata.items.append(
        compute_v1.Items(key="startup-script", value=user_data_script)
    )

    # 6. Build Instance Resource Definition
    instance = compute_v1.Instance()
    instance.name = instance_name
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"
    instance.disks = [boot_disk, data_disk]
    instance.guest_accelerators = guest_accelerators
    instance.network_interfaces = [network_interface]
    instance.metadata = metadata

    # Scheduling settings required for GPUs on GCE
    if guest_accelerators:
        scheduling = compute_v1.Scheduling()
        scheduling.on_host_maintenance = compute_v1.Scheduling.OnHostMaintenance.TERMINATE.name
        instance.scheduling = scheduling

    # 7. Issue API Call to Create Instance
    operation = instance_client.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance
    )

    print("Waiting for instance creation operation to complete...")
    operation.result()  # Wait for API completion

    # Retrieve instance details (Public IP)
    created_instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)
    public_ip = created_instance.network_interfaces[0].access_configs[0].nat_i_p

    print(f"\n=== GCP GPU Instance Provisioned ===")
    print(f"Instance Name: {instance_name}")
    print(f"Public IP:     {public_ip}")
    print(f"SSH Command:   gcloud compute ssh {instance_name} --zone={zone}")
    print("Bootstrap script is running in the background. Monitor logs via:")
    print(f"  gcloud compute ssh {instance_name} --zone={zone} -- 'tail -f /var/log/startup-script.log'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCP Compute Engine GPU Instance Provisioning Script")
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud Project ID")
    parser.add_argument("--zone", type=str, default="us-central1-a", help="GCP Zone (e.g. us-central1-a)")
    parser.add_argument("--instance_name", type=str, default="cbramod-training-worker", help="VM Instance Name")
    parser.add_argument("--machine_type", type=str, default="g2-standard-8", help="GCP Machine Type (e.g. g2-standard-8, n1-standard-8)")
    parser.add_argument("--accelerator_type", type=str, default="nvidia-l4", help="GPU Type (e.g., nvidia-l4, nvidia-tesla-t4, nvidia-a100-80gb)")
    parser.add_argument("--accelerator_count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--bucket", type=str, required=True, help="Target GCS / Object Store Bucket")
    parser.add_argument("--s3_prefix", type=str, required=True, help="Dataset folder prefix on S3/GCS")
    parser.add_argument("--mount_point", type=str, default="/mnt/nvme/cbra_data", help="Local block storage mount point")
    parser.add_argument("--repo_url", type=str, default="", help="Git repository URL to clone")

    args = parser.parse_args()

    check_prerequisites()

    bootstrap_script = generate_cloud_init_script(
        bucket_name=args.bucket,
        s3_prefix=args.s3_prefix,
        mount_point=args.mount_point,
        git_repo_url=args.repo_url
    )

    deploy_gcp_compute_instance(
        project_id=args.project_id,
        zone=args.zone,
        instance_name=args.instance_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        user_data_script=bootstrap_script
    )