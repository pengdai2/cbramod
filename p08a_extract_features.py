"""
Standalone backbone feature extraction for CBraMod's probe-training pipeline (p08b_finetune_probing.py).

Runs the frozen CBraMod backbone once over EVERY accepted subject in a p03-generated
master_manifest.csv (not just one split), caching pooled embeddings + labels + subject_ids + stages +
indices to a single .pt file. p08b then carves out train/val (or any CV fold's) subject subset from
THIS SAME cache via CachedFeatureSubjectDataset's subject filter -- no separate extraction per split,
and no re-extraction across repeated training runs/hyperparameter sweeps.

This is a standalone data-prep stage (the same category as p01/p02/p03), not a training concern, on
purpose: the master cache is a reusable pipeline artifact that p08b (or anything else) can point at,
and its own CLI surface (manifest/data-dir/filter-stage/model dims/cache paths) has nothing to do with
probe-training hyperparameters (epochs/head-lr/imbalance-strategy/etc.), so neither parser is cluttered
with flags irrelevant to what it does.

Usage:
  python p08a_extract_features.py \
      --master-manifest /data/eeg_study/master_manifest.csv \
      --data-dir /data/eeg_study/npy_files \
      --filter-stage N2,N3 \
      --cache-dir /data/eeg_study/cache \
      --num-workers 8
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from tqdm import tqdm

from cbramod_common import CBraModFeatureExtractor, PANSleepEEGDataset, setup_common_cli_parser, seed_everything
from cbramod_utils import setup_logger
from torch.utils.data import DataLoader


def extract_and_cache(config: argparse.Namespace, logger, manifest_path: Path, output_cache_path: Path) -> None:
    """Reads .npy files, applies stage filtering, extracts backbone embeddings, and caches feats + labels + subject_ids."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filter_str = f" [Filter: {config.filter_stage}]" if config.filter_stage else ""
    logger.info(f"Initializing PANSleepEEGDataset from: {manifest_path}{filter_str}")

    dataset = PANSleepEEGDataset(
        manifest_csv=manifest_path,
        data_dir=config.data_dir,
        filter_stage=config.filter_stage
    )
    logger.info(f"Successfully indexed {len(dataset):,} valid stage-filtered window references.")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    logger.info(f"Extracting backbone representations to: {output_cache_path}")
    extractor = CBraModFeatureExtractor(
        num_channels=config.num_channels,
        sfreq=config.sfreq
    ).to(device)
    extractor.eval()

    all_embeddings, all_labels, all_subject_ids, all_stages, all_indices = [], [], [], [], []
    start_time = time.time()

    use_amp = not config.no_amp
    with torch.no_grad():
        for batch_x, batch_y, batch_subj, batch_stg, batch_idx in tqdm(loader, desc="Extracting", unit="batch"):
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                pooled_feats = extractor(batch_x)

            all_embeddings.append(pooled_feats.cpu().float())
            all_labels.append(batch_y.cpu())
            all_subject_ids.extend(batch_subj)
            all_stages.extend(batch_stg)
            all_indices.extend(batch_idx)

    cached_feats = torch.cat(all_embeddings, dim=0)
    cached_labels = torch.cat(all_labels, dim=0)

    torch.save({
        "feats": cached_feats,
        "labels": cached_labels,
        "subject_ids": all_subject_ids,
        "stages": all_stages,
        "indices": all_indices
    }, output_cache_path)

    del extractor, dataset, loader, all_embeddings, all_labels, all_subject_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    file_size_mb = output_cache_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"✓ Extraction complete ({elapsed:.1f}s) | Subjects: {len(set(all_subject_ids)):,} | "
        f"Windows: {len(cached_feats):,} | Cache Size: {file_size_mb:.2f} MB"
    )


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone CBraMod backbone feature extraction over a p03 master_manifest.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    setup_common_cli_parser(parser)

    cache_group = parser.add_argument_group("Extraction & Cache")
    cache_group.add_argument(
        "--master-manifest", type=str, required=True,
        help="Path to p03's master_manifest.csv (every accepted subject, regardless of split)."
    )
    cache_group.add_argument("--cache-dir", type=str, required=True, help="Directory to write the cached embeddings file to.")
    cache_group.add_argument(
        "--master-cache-name", type=str, default="cached_master_embeddings.pt",
        help="Filename for the whole-cohort cached embeddings file."
    )
    cache_group.add_argument("--force-extract", action="store_true", help="Re-extract even if the cache file already exists.")

    return parser.parse_args()


def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cache_dir / "p08a_extract_features.log")

    master_cache_path = cache_dir / args.master_cache_name
    if master_cache_path.exists() and not args.force_extract:
        logger.info(f"Cache already exists at {master_cache_path}. Use --force-extract to re-extract. Nothing to do.")
        return

    extract_and_cache(args, logger, Path(args.master_manifest), master_cache_path)


if __name__ == "__main__":
    main()
