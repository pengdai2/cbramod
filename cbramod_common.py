import argparse
from collections import defaultdict
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import safetensors
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from einops.layers.torch import Rearrange
from braindecode.models import CBraMod
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score
)


# =====================================================================
# DATASETS
# =====================================================================

def parse_filter(
    filter: Optional[Union[str, List[str], Set[str], Tuple[str, ...]]]
) -> Optional[Set[str]]:
    """Standardizes optional filter formats into a set of strings."""
    if isinstance(filter, str):
        return {s.strip() for s in filter.split(",")}
    elif isinstance(filter, (list, tuple, set)):
        return {str(s).strip() for s in filter}
    return None


def extract_valid_window_indices(
    meta_path: Optional[Path],
    npy_path: Path,
    filter_stage: Optional[Set[str]] = None,
    memory_map: bool = True
) -> Tuple[List[int], List[str], int]:
    """
    Parses subject metadata to extract valid window row indices matching 
    quality and stage filter criteria.
    
    Returns:
        Tuple of (valid_window_indices, excluded_window_count)
    """
    valid_indices = []
    excluded_count = 0

    if meta_path and meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            stages_list = meta.get("stages", [])

            for slice_info in meta.get("slices", []):
                # Check slice validity
                if not slice_info.get("is_valid", True):
                    excluded_count += 1
                    continue

                window_idx = slice_info["window_idx"]  # Direct row index in .npy array

                # Resolve sleep stage for window
                slice_stage = slice_info.get("stage")
                if slice_stage is None and window_idx < len(stages_list):
                    slice_stage = stages_list[window_idx]

                # Apply sleep stage filtering
                if filter_stage is not None and slice_stage not in filter_stage:
                    excluded_count += 1
                    continue

                valid_indices.append(window_idx)

            return valid_indices, np.array(stages_list)[valid_indices].tolist(), excluded_count
        except Exception as e:
            print(f"  [Warning] Failed to parse meta JSON {meta_path}: {e}. Falling back to array bounds.")

    # Fallback if meta_path is missing or invalid
    try:
        if memory_map:
            mmap_data = np.load(npy_path, mmap_mode="r")
            actual_slices = mmap_data.shape[0]
        else:
            data = np.load(npy_path)
            actual_slices = data.shape[0]

        return list(range(actual_slices)), [None] * actual_slices, 0
    except Exception as e:
        print(f"  [Warning] Failed to read array shape from {npy_path}: {e}")
        return [], [], 0


class PANSubjectEEGDataset(Dataset):
    """
    PyTorch Dataset that loads subject recordings from manifest CSVs.
    Returns all valid windows for a subject in a single tensor.
    """
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        filter_subject: Optional[List[str]] = None,
        filter_stage: Optional[Union[str, List[str], Set[str], Tuple[str, ...]]] = None,
        memory_map: bool = True
    ):
        self.manifest_csv = Path(manifest_csv)
        self.data_dir = Path(data_dir) if data_dir else None
        self.memory_map = memory_map
        self.filter_subject = parse_filter(filter_subject)
        self.filter_stage = parse_filter(filter_stage)

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_csv}")

        df = pd.read_csv(self.manifest_csv)
        if self.filter_subject:
            target_ids = set(self.filter_subject)
            df["subject_id"] = df["subject_id"].astype(str)
            df = df[df["subject_id"].isin(target_ids)].copy()
        self.df = df

        self.subjects: List[Tuple[str, Path, List[int], int, List[str]]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        print(f"Indexing subject recordings from manifest: {self.manifest_csv.name}...")
        total_valid_windows = 0
        total_skipped_slices = 0
        skipped_subjects = 0

        for _, row in self.df.iterrows():
            subject_id = str(row["subject_id"])
            raw_npy_path = Path(row["npy_path"])
            raw_meta_path = Path(row["meta_path"]) if "meta_path" in row and pd.notna(row["meta_path"]) else None
            label = int(row["label"])

            if label == -1:
                print(f"  [Warning] Subject {subject_id} missing label, skipping...")
                skipped_subjects += 1
                continue

            if self.data_dir:
                npy_path = self.data_dir / raw_npy_path if not raw_npy_path.is_absolute() else raw_npy_path
                meta_path = self.data_dir / raw_meta_path if raw_meta_path and not raw_meta_path.is_absolute() else raw_meta_path
            else:
                npy_path = raw_npy_path
                meta_path = raw_meta_path

            if not npy_path.exists():
                print(f"  [Warning] Subject {subject_id} missing tensor file: {npy_path}, skipping...")
                skipped_subjects += 1
                continue

            valid_indices, stages, excluded_count = extract_valid_window_indices(
                meta_path=meta_path,
                npy_path=npy_path,
                filter_stage=self.filter_stage,
                memory_map=self.memory_map
            )

            if not valid_indices:
                print(f"  [Warning] Subject {subject_id} has 0 valid windows after filtering, skipping...")
                skipped_subjects += 1
                continue

            self.subjects.append((subject_id, npy_path, valid_indices, label, stages))
            total_valid_windows += len(valid_indices)
            total_skipped_slices += excluded_count

        filter_stage_info = f" [Filter Stage: {','.join(sorted(self.filter_stage))}]" if self.filter_stage else ""
        print(
            f"  -> Indexing complete{filter_stage_info}: {len(self.subjects)} subjects loaded "
            f"({total_valid_windows:,} total valid windows, {total_skipped_slices:,} excluded/filtered windows, "
            f"{skipped_subjects} subjects skipped)."
        )

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        subject_id, npy_path, valid_indices, label, stages = self.subjects[idx]

        if self.memory_map:
            mmap_data = np.load(npy_path, mmap_mode="r")
            subj_data = np.array(mmap_data[valid_indices], dtype=np.float32)
        else:
            data = np.load(npy_path)
            subj_data = data[valid_indices].astype(np.float32)

        x_tensor = torch.from_numpy(subj_data)
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor, subject_id, stages, valid_indices


class PANSleepEEGDataset(Dataset):
    """
    PyTorch Dataset returning individual 30s window slices.
    """
    def __init__(
        self,
        manifest_csv: Path,
        data_dir: Optional[Union[str, Path]] = None,
        filter_subject: Optional[List[str]] = None, # XXX / TODO: subject filtering
        filter_stage: Optional[Union[str, List[str], Set[str], Tuple[str, ...]]] = None,
        memory_map: bool = True
    ):
        self.manifest_csv = Path(manifest_csv)
        self.data_dir = Path(data_dir) if data_dir else None
        self.memory_map = memory_map
        self.filter_subject = parse_filter(filter_subject)
        self.filter_stage = parse_filter(filter_stage)

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_csv}")

        df = pd.read_csv(self.manifest_csv)
        if self.filter_subject:
            target_ids = set(self.filter_subject)
            df["subject_id"] = df["subject_id"].astype(str)
            df = df[df["subject_id"].isin(target_ids)].copy()
        self.df = df

        self.samples: List[Tuple[Path, int, int, str]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        print(f"Indexing samples from manifest: {self.manifest_csv.name}...")
        total_valid_slices = 0
        total_skipped_slices = 0

        for _, row in self.df.iterrows():
            subject_id = str(row["subject_id"])
            raw_npy_path = Path(row["npy_path"])
            raw_meta_path = Path(row["meta_path"]) if "meta_path" in row and pd.notna(row["meta_path"]) else None
            label = int(row["label"])

            if label == -1:
                print(f"  [Warning] Subject {subject_id} missing label, skipping...")
                continue

            if self.data_dir:
                npy_path = self.data_dir / raw_npy_path if not raw_npy_path.is_absolute() else raw_npy_path
                meta_path = self.data_dir / raw_meta_path if raw_meta_path and not raw_meta_path.is_absolute() else raw_meta_path
            else:
                npy_path = raw_npy_path
                meta_path = raw_meta_path

            if not npy_path.exists():
                print(f"  [Warning] Subject {subject_id} missing tensor file: {npy_path}, skipping...")
                continue

            valid_indices, stages, excluded_count = extract_valid_window_indices(
                meta_path=meta_path,
                npy_path=npy_path,
                filter_stage=self.filter_stage,
                memory_map=self.memory_map
            )
            total_skipped_slices += excluded_count

            # `stages` from extract_valid_window_indices is already compacted
            # to align positionally with `valid_indices` (stages[i] <->
            # valid_indices[i]), not indexed by the raw window index, so we
            # must look it up by position (i), not by w_idx itself.
            for i, w_idx in enumerate(valid_indices):
                self.samples.append((npy_path, w_idx, label, subject_id, stages[i]))
                total_valid_slices += 1

        filter_info = f" [Filter Stage: {','.join(sorted(self.filter_stage))}]" if self.filter_stage else ""
        print(
            f"  -> Indexing complete{filter_info}: {total_valid_slices:,} valid windows loaded "
            f"across {len(self.df)} subjects ({total_skipped_slices:,} excluded/filtered windows)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        npy_path, window_idx, label, subject_id, stage = self.samples[idx]

        if self.memory_map:
            data = np.load(npy_path, mmap_mode="r")
            slice_data = np.array(data[window_idx], dtype=np.float32)
        else:
            data = np.load(npy_path)
            slice_data = data[window_idx].astype(np.float32)

        x_tensor = torch.from_numpy(slice_data)
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor, subject_id, stage, window_idx


class CachedFeatureSubjectDataset(Dataset):
    """Dataset that groups pre-extracted features by subject ID for patient-level inference."""
    def __init__(
        self,
        pt_path: Union[str, Path],
        filter_subject: Optional[List[str]] = None
    ):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        # "stages"/"indices" should always be present -- extract_and_cache()
        # (p08b_finetune_probing.py) writes them unconditionally. Fail loudly
        # with an actionable message rather than silently defaulting to
        # None/range placeholders: a cache missing this metadata means the
        # extraction run that produced it either predates stage tracking or
        # went through a different/broken pipeline, and callers that rely on
        # per-window stage (e.g. p09c's tier reports) would otherwise get
        # quietly wrong results instead of a clear signal to re-extract.
        missing_keys = [k for k in ("stages", "indices") if k not in data]
        if missing_keys:
            raise KeyError(
                f"Cached feature file '{pt_path}' is missing key(s) {missing_keys}. "
                "extract_and_cache() always writes these alongside feats/labels/subject_ids, "
                "so this cache was likely produced by a stale extraction run (predating stage "
                "tracking) or a different pipeline entirely. Re-run feature extraction to "
                "regenerate this cache rather than proceeding without stage/index metadata."
            )

        self.feats = data["feats"]
        self.labels = data["labels"]
        # Wrapped in np.array (rather than left as the plain lists that
        # extract_and_cache() saves) so the boolean-mask indexing in
        # __getitem__ works.
        self.stages = np.array(data["stages"], dtype=object)
        self.indices = np.array(data["indices"])
        self.subject_ids = np.array(data["subject_ids"])
        self.unique_subjects = np.unique(self.subject_ids)
        self.filter_subject = parse_filter(filter_subject)
        if self.filter_subject:
            self.unique_subjects = self.unique_subjects[np.isin(self.unique_subjects, list(self.filter_subject))]

    def __len__(self) -> int:
        return len(self.unique_subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, np.ndarray, np.ndarray]:
        subject_id = self.unique_subjects[idx]
        mask = (self.subject_ids == subject_id)
        subj_feats = self.feats[mask]
        subj_stages = self.stages[mask]
        subj_indices = self.indices[mask]

        # All windows for a given subject share the same patient-level ground
        # truth. Kept as a tensor (not .item()'d to a plain int) for
        # consistency with PANSubjectEEGDataset.__getitem__'s y_tensor --
        # callers that treat both datasets generically can rely on this slot
        # always being a torch.Tensor regardless of which dataset produced it.
        subj_label = self.labels[mask][0]

        return subj_feats, subj_label, subject_id, subj_stages, subj_indices


class SyntheticEEGDataset(torch.utils.data.Dataset):
    """
    Generates synthetic EEG tensors matching CBraMod input specs for pipeline verification:
    Shape: [Batch, Channels, Time_Samples] -> [B, 64, 6000] (30s @ 200 Hz)
    """
    def __init__(self, num_samples: int = 128, channels: int = 64, time_samples: int = 6000, num_classes: int = 2):
        self.num_samples = num_samples
        # Generate random Gaussian noise with synthetic 12 Hz sinusoidal bursts (simulated spindles)
        self.data = torch.randn(num_samples, channels, time_samples, dtype=torch.float32)

        # Inject synthetic 12 Hz sine wave in central channels for half the batch
        t = torch.linspace(0, 30, time_samples)
        spindle_wave = 2.0 * torch.sin(2 * np.pi * 12 * t)
        for i in range(num_samples // 2):
            self.data[i, :4, 2000:2400] += spindle_wave[2000:2400] # Inject 2-second burst

        self.labels = torch.cat([torch.ones(num_samples // 2, dtype=torch.long),
                                 torch.zeros(num_samples - num_samples // 2, dtype=torch.long)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# =====================================================================
# CLASSIFICATION HEAD ARCHITECTURES
# =====================================================================

class LinearProbeHead(nn.Module):
    """
    1-Layer True Linear Classification Head.
    Flattens spatial/temporal embeddings and projects directly to output logits.
    """
    def __init__(
        self,
        num_patches: int = 30,
        emb_dim: int = 200,
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__()
        in_features = num_patches * emb_dim

        self.head = nn.Sequential(
            Rearrange("b s p -> b (s p)"),
            nn.LayerNorm(in_features),  # Standardizes embedding drift per sample
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MLPProbeHead(nn.Module):
    """
    2-Layer Non-Linear MLP Head with LayerNorm and Dropout.
    """
    def __init__(
        self,
        num_patches: int = 30,
        emb_dim: int = 200,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        in_features = num_patches * emb_dim

        self.head = nn.Sequential(
            Rearrange("b s p -> b (s p)"),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CBraModE2EClassifier(nn.Module):
    """
    Unified End-to-End Architecture wrapping the CBraMod backbone
    and a configurable probing head (Linear or MLP) into a single module.
    """
    def __init__(
        self,
        num_channels: int = 64,
        sfreq: float = 200.0,
        num_patches: int = 30,
        emb_dim: int = 200,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        head_type: str = "linear"
    ):
        super().__init__()
        self.backbone = CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained",
            n_chans=num_channels,
            sfreq=sfreq,
            return_encoder_output=True
        )

        head_type_lower = head_type.lower()
        if head_type_lower == "linear":
            self.head = LinearProbeHead(
                num_patches=num_patches,
                emb_dim=emb_dim,
                num_classes=num_classes
            )
        elif head_type_lower == "mlp":
            self.head = MLPProbeHead(
                num_patches=num_patches,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid head_type: '{head_type}'. Choose 'linear' or 'mlp'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone output shape: [Batch, Channels, Patches, EmbDim]
        feats = self.backbone(x).mean(dim=1)  # Spatial channel mean -> [Batch, Patches, EmbDim]
        return self.head(feats)


class CBraModFeatureExtractor(nn.Module):
    """Backbone extractor that channel-pools [B, C, S, P] -> [B, S, P]."""
    def __init__(self, num_channels: int = 64, sfreq: float = 200.0):
        super().__init__()
        self.backbone = CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained",
            n_chans=num_channels,
            sfreq=sfreq,
            return_encoder_output=True
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats.mean(dim=1)


def compute_pooled_scores(
    window_probs: np.ndarray,
    method: str = "p85_score",
    top_percentile: float = 0.10,
    t_window: float = 0.60
) -> Union[float, np.ndarray]:
    """
    Aggregates window-level probabilities into subject-level class scores.
    Supports both 1D arrays (binary positive class probabilities) and 
    2D arrays (shape: [Num_Windows, Num_Classes]).

    Args:
        window_probs: Array of probabilities [N] or [N, K].
        method: Pooling strategy ('p85_score', 'top_10_mean', 'trimmed_top_10', 'burden_ratio').
        top_percentile: Fraction of top windows to evaluate (default: 0.10).
        t_window: Window-level probability threshold for burden ratio.

    Returns:
        float if 1D input (binary), or np.ndarray [K] if 2D input (multi-class).
    """
    if len(window_probs) == 0:
        return 0.0 if window_probs.ndim == 1 else np.array([])

    is_1d = (window_probs.ndim == 1)
    N = len(window_probs)
    k_len = max(1, int(np.ceil(N * top_percentile)))

    if method == "p85_score":
        if is_1d:
            return float(np.percentile(window_probs, 85))
        return np.percentile(window_probs, 85, axis=0)

    elif method == "top_10_mean":
        if is_1d:
            sorted_p = np.sort(window_probs)[::-1]
            return float(np.mean(sorted_p[:k_len]))
        sorted_p = np.sort(window_probs, axis=0)[::-1, :]
        return np.mean(sorted_p[:k_len, :], axis=0)

    elif method == "trimmed_top_10":
        skip = int(N * 0.02)
        if is_1d:
            sorted_p = np.sort(window_probs)[::-1]
            return float(np.mean(sorted_p[skip : skip + k_len]))
        sorted_p = np.sort(window_probs, axis=0)[::-1, :]
        return np.mean(sorted_p[skip : skip + k_len, :], axis=0)

    elif method == "burden_ratio":
        if is_1d:
            return float(np.mean(window_probs >= t_window))
        # Multi-class burden: proportion of windows where class k is argmax and >= t_window
        dominant_class = np.argmax(window_probs, axis=1)
        K = window_probs.shape[1]
        scores = np.zeros(K, dtype=np.float64)
        for c in range(K):
            scores[c] = np.mean((dominant_class == c) & (window_probs[:, c] >= t_window))
        return scores

    else:
        raise ValueError(f"Unsupported pooling method: {method}")


def compute_leave_one_out_contributions(
    window_probs: np.ndarray,
    method: str = "p85_score",
    top_percentile: float = 0.10,
    t_window: float = 0.60
) -> np.ndarray:
    """
    For each window i, computes contribution_i = full_pooled_score -
    pooled_score_with_window_i_removed -- how much removing that single
    window would change the subject-level pooled score under the ACTIVE
    pooling formula. Positive means the window was pulling the pooled score
    up (removing it lowers the score); negative means the opposite.

    This answers an exact, retraining-free question -- "which windows does
    the pooling formula itself say matter for this subject's score" --
    which is logically prior to, and distinct from, "why did that window's
    own classifier output score high" (the attribution/morphology
    question). Binary (1D positive-class-probability) inputs only; 2D
    multi-class arrays aren't supported here.

    Cost is O(N^2) (pooling is recomputed from scratch per left-out window),
    but N = windows per subject is small (tens to low hundreds), so this is
    cheap in practice -- no need for Monte Carlo Shapley approximation.
    """
    window_probs = np.asarray(window_probs, dtype=np.float64)
    n = len(window_probs)
    if n <= 1:
        return np.zeros(n)

    full_score = compute_pooled_scores(window_probs, method=method, top_percentile=top_percentile, t_window=t_window)

    contributions = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        loo_score = compute_pooled_scores(window_probs[mask], method=method, top_percentile=top_percentile, t_window=t_window)
        contributions[i] = full_score - loo_score
        mask[i] = True

    return contributions


def rank_leave_one_out_contributions(
    window_probs: np.ndarray,
    method: str = "p85_score",
    top_percentile: float = 0.10,
    t_window: float = 0.60,
    top_k: Optional[int] = None
) -> List[Tuple[int, float]]:
    """
    Convenience wrapper around `compute_leave_one_out_contributions`:
    returns (window_index, contribution) pairs sorted by |contribution|
    descending, optionally truncated to the top_k largest-magnitude
    contributors. `window_index` is positional within `window_probs` --
    callers need their own raw-index array (e.g. a subject's
    `report["indices"]` from p09c) to map back to actual .npy row numbers.
    """
    contributions = compute_leave_one_out_contributions(
        window_probs, method=method, top_percentile=top_percentile, t_window=t_window
    )
    order = np.argsort(np.abs(contributions))[::-1]
    if top_k is not None:
        order = order[:top_k]
    return [(int(i), float(contributions[i])) for i in order]


def setup_data_loader_and_criterion(
    dataset: Dataset,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    imbalance_strategy: str,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[DataLoader, nn.Module]:
    """
    Configures training DataLoader and CrossEntropyLoss based on imbalance strategy.

    Strategies:
      - 'sampler': Uses WeightedRandomSampler to balance batch draws. Unweighted CrossEntropyLoss.
      - 'loss_weights': Standard DataLoader with shuffle=True. Inverse class frequency weighted CrossEntropyLoss.
      - 'none': Standard DataLoader with shuffle=True and unweighted CrossEntropyLoss.
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    logger.info("--- Training Set Class Distribution ---")
    for class_id, count in enumerate(class_counts):
        pct = (count / total_samples) * 100.0 if total_samples > 0 else 0.0
        logger.info(f"  Class {class_id}: {count:,} samples ({pct:.2f}%)")

    persistent_workers = num_workers > 0

    if imbalance_strategy == "sampler":
        logger.info("--> Imbalance Strategy: WeightedRandomSampler (Resampling minority class batches)")
        class_weights_raw = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights_raw[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent_workers
        )
        criterion = nn.CrossEntropyLoss()

    elif imbalance_strategy == "loss_weights":
        logger.info("--> Imbalance Strategy: Class-Weighted CrossEntropyLoss")
        balanced_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
        weights_tensor = torch.from_numpy(balanced_weights).float().to(device)

        weight_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(balanced_weights)])
        logger.info(f"    Derived Loss Weights: [{weight_str}]")

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent_workers
        )

    elif imbalance_strategy == "none":
        logger.info("--> Imbalance Strategy: None (Unweighted training)")
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent_workers
        )

    else:
        raise ValueError(f"Invalid imbalance strategy: {imbalance_strategy}. Choose 'sampler', 'loss_weights', or 'none'.")

    return train_loader, criterion


class CBraModTrainer:
    """Base class for training CBraMod models with subject-level evaluation."""
    def __init__(self, config: argparse.Namespace, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_subject_pooling(
        self, 
        val_probs: np.ndarray, 
        val_targets: np.ndarray, 
        val_subject_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Groups window probabilities by subject using an O(N) pre-indexed map, 
        applies all 4 pooling strategies, and performs threshold tuning.
        """
        strategies = ["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"]

        # O(N) linear indexing pass to pre-group window indices by subject ID
        subj_to_indices = defaultdict(list)
        for idx, subj in enumerate(val_subject_ids):
            subj_to_indices[subj].append(idx)

        subject_data = {strat: [] for strat in strategies}
        subject_labels = []

        # Iterate through pre-grouped subject slices
        for subj, idxs in subj_to_indices.items():
            idx_arr = np.array(idxs, dtype=np.int64)
            subj_probs = val_probs[idx_arr]
            subj_gt = val_targets[idx_arr[0]]
            subject_labels.append(subj_gt)

            for strat in strategies:
                if self.config.num_classes == 2:
                    score = compute_pooled_scores(
                        subj_probs[:, 1], 
                        method=strat, 
                        top_percentile=self.config.top_percentile, 
                        t_window=self.config.t_window
                    )
                else:
                    score = compute_pooled_scores(
                        subj_probs, 
                        method=strat, 
                        top_percentile=self.config.top_percentile, 
                        t_window=self.config.t_window
                    )
                subject_data[strat].append(score)

        subject_labels = np.array(subject_labels)
        results = {}

        # Strategy evaluation and threshold optimization
        for strat in strategies:
            scores = np.array(subject_data[strat])
            
            if self.config.num_classes == 2:
                # Binary threshold sweep to maximize Macro F1 on Subject predictions
                best_t = 0.5
                best_f1 = 0.0
                thresholds = np.linspace(0.01, 0.99, 99)
                
                for t in thresholds:
                    preds = (scores >= t).astype(int)
                    f1 = f1_score(subject_labels, preds, average="macro", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
                
                final_preds = (scores >= best_t).astype(int)
                acc = accuracy_score(subject_labels, final_preds)
                sensitivity = recall_score(subject_labels, final_preds)
                specificity = recall_score(subject_labels, final_preds, pos_label=0)
                roc_auc = roc_auc_score(subject_labels, scores) if len(np.unique(subject_labels)) > 1 else 0.5

                results[strat] = {
                    "subject_macro_f1": best_f1,
                    "optimal_threshold": float(best_t),
                    "subject_accuracy": acc,
                    "subject_sensitivity": sensitivity,
                    "subject_specificity": specificity,
                    "roc_auc": roc_auc
                }
            else:
                # Multi-class argmax selection
                preds = np.argmax(scores, axis=1)
                macro_f1 = f1_score(subject_labels, preds, average="macro", zero_division=0)
                acc = accuracy_score(subject_labels, preds)
                results[strat] = {
                    "subject_macro_f1": macro_f1,
                    "optimal_threshold": 0.5,
                    "subject_accuracy": acc,
                    "roc_auc": 0.5
                }

        return results  

    def train(self, train_path: Path, val_path: Path) -> dict:
        pass  # Placeholder for training loop implementation  


def setup_common_cli_parser(parser: argparse.ArgumentParser) -> None:
    # CBraMod Architecture Controls    
    cbra_group = parser.add_argument_group("CBraMod Architecture Controls")
    cbra_group.add_argument("--batch-size", type=int, default=512, help="[B]atch size for model execution")
    cbra_group.add_argument("--num-channels", type=int, default=64, help="EEG [C]hannel count")
    cbra_group.add_argument("--num-patches", type=int, default=30, help="Number of temporal patches in a [S]egment")
    cbra_group.add_argument("--sfreq", type=float, default=200.0, help="Number of EEG samples in a [P]atch")
    cbra_group.add_argument("--cbra-dim", type=int, default=200, help="CBraMod embedding dimension per patch")
    cbra_group.add_argument("--head-type", type=str, choices=["linear", "mlp"], default="mlp", help="Classification head architecture: 'linear' (1-layer) or 'mlp' (2-layer)")
    cbra_group.add_argument("--head-dim", type=int, default=128, help="Head dimension for 2 layer MLP")
    cbra_group.add_argument("--num-classes", type=int, default=2, help="Number of target classes")
    cbra_group.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in head")

    # Data and Filtering Controls
    data_group = parser.add_argument_group("Data and Filtering")
    data_group.add_argument("--data-dir", type=str, default=None, help="Root directory containing .npy files")
    data_group.add_argument("--filter-stage", type=str, default="N2,N3", help="Comma-separated sleep stages to pass into PANSleepEEGDataset (e.g., N2,N3)")

    # System Controls
    sys_group = parser.add_argument_group("System Controls")
    sys_group.add_argument("--num-workers", type=int, default=4, help="DataLoader CPU workers for disk reads")
    sys_group.add_argument("--seed", type=int, default=42, help="Random seed for deterministic execution")
    sys_group.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")


def setup_training_cli_parser(
    description: str = "CBraMod Training"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    setup_common_cli_parser(parser)

    # Training Manifests
    manifest_group = parser.add_argument_group("Training Manifests")
    manifest_group.add_argument("--train-manifest", type=str, default=None, help="Path to training manifest file (CSV/TSV/JSON)")
    manifest_group.add_argument("--val-manifest", type=str, default=None, help="Path to validation manifest file (CSV/TSV/JSON)")

    # Checkpoint Output
    ckpt_group = parser.add_argument_group("Checkpoint")
    ckpt_group.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save the checkpoint")
    ckpt_group.add_argument("--checkpoint-filename", type=str, default="cbramod_ckpt.pt", help="Filename for checkpoint")

    # Pooling Strategy
    pool_group = parser.add_argument_group("Pooling Strategy")
    pool_group.add_argument(
        "--primary-pooling",
        type=str,
        default="p85_score",
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio"],
        help="Primary pooling strategy used for early stopping and model selection"
    )
    pool_group.add_argument("--top-percentile", type=float, default=0.10, help="Top percentile ratio for top-K pooling methods")
    pool_group.add_argument("--t-window", type=float, default=0.60, help="Window threshold for pathology burden ratio")

    # Hyperparameters
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument("--epochs", type=int, default=40, help="Maximum training epochs for linear probe head")
    hp_group.add_argument("--head-lr", type=float, default=1e-4, help="Initial learning rate for classification head")
    hp_group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing scheduler")
    hp_group.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay regularizer")
    hp_group.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without Subject F1 improvement)")
    hp_group.add_argument(
        "--imbalance-strategy",
        type=str,
        choices=["sampler", "loss_weights", "none"],
        default="loss_weights",
        help="Class imbalance handling: 'sampler' (WeightedRandomSampler), 'loss_weights' (Class-Weighted CrossEntropy), or 'none'"
    )

    return parser


def setup_inference_cli_parser(
    description: str = "CBraMod Inference"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    setup_common_cli_parser(parser)

    # Checkpoint Input
    ckpt_group = parser.add_argument_group("Checkpoint")
    ckpt_group.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # Subject Data
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--manifest", type=str, help="Path to test_manifest.csv for raw .npy inference")
    data_group.add_argument("--features-pt", type=str, help="Path to pre-extracted test features (.pt)")

    # Subject Filtering
    subject_group = parser.add_argument_group("Subject Filtering")
    subject_group.add_argument("--subject-id", type=str, default=None, help="Optional comma-separated list of specific Subject IDs to analyze (e.g., GRINS0322,GRINS0038).")

    # Pooling Strategy
    # Defaults are left as None (rather than a hardcoded value) so
    # resolve_pooling_config() can tell "user didn't pass this flag" apart
    # from "user explicitly chose the same value the checkpoint already
    # has" -- the former falls back to whatever pooling config was saved in
    # the checkpoint at training time, the latter always wins.
    pool_group = parser.add_argument_group("Pooling Strategy")
    pool_group.add_argument(
        "--pooling-strategy",
        type=str,
        default=None,
        choices=["p85_score", "top_10_mean", "trimmed_top_10", "burden_ratio", "all"],
        help="Pooling strategy choice, or 'all' for full comparative report. "
             "Default: the primary_pooling strategy saved in the checkpoint at training time "
             "(falls back to 'p85_score' if the checkpoint has none)."
    )
    pool_group.add_argument(
        "--top-percentile", type=float, default=None,
        help="Top percentile ratio for top-K pooling methods. "
             "Default: the value saved in the checkpoint at training time (falls back to 0.10)."
    )
    pool_group.add_argument(
        "--t-window", type=float, default=None,
        help="Window threshold for burden ratio pooling. "
             "Default: the value saved in the checkpoint at training time (falls back to 0.60)."
    )

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--override-threshold", type=float, default=None, help="Override operating decision threshold")
    misc_group.add_argument("--output-dir", type=str, default=None, help="Output directory for the subject analysis")

    return parser


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, dict, Union[int, str], Dict[str, Union[str, float]]]:
    """
    Loads checkpoint weights into the model architecture.
    Handles both head-only (backbone frozen / LP-FT) state dicts and full model state dicts.

    Also surfaces the pooling configuration ("primary_pooling", "top_percentile",
    "t_window") saved alongside the weights at training time, so downstream
    inference/analysis scripts can reproduce the exact pooling that produced
    the checkpoint's calibrated thresholds by default -- see
    `resolve_pooling_config`. Only keys actually present in the checkpoint are
    included, so older checkpoints saved before this field existed degrade
    gracefully to an empty dict.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint state dict not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract state dict dict structure if wrapped inside checkpoint metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    optimal_thresholds = checkpoint.get("optimal_thresholds", {}) if isinstance(checkpoint, dict) else {}
    epoch = checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"
    ckpt_pooling_params = {
        key: checkpoint[key]
        for key in ("primary_pooling", "top_percentile", "t_window")
        if isinstance(checkpoint, dict) and key in checkpoint
    }

    # Strategy 1: Attempt direct full-model state dict load (Full Fine-Tuning)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded full model checkpoint (strict=True) from epoch {epoch}.")
        return model, optimal_thresholds, epoch, ckpt_pooling_params
    except Exception:
        pass

    # Strategy 2: Attempt head-only state dict load into model.head (Linear Probe / Head Frozen)
    head_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone.") and not k.startswith("encoder."):
            head_state_dict[k] = v

    if hasattr(model, "head") and head_state_dict:
        try:
            model.head.load_state_dict(head_state_dict, strict=True)
            print(f"Successfully loaded head-only state dict into model.head from epoch {epoch}.")
            return model, optimal_thresholds, epoch, ckpt_pooling_params
        except Exception:
            pass

    # Strategy 3: Fallback load with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with strict=False from epoch {epoch}.")
    if missing_keys:
        print(f"  [Info] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  [Info] Unexpected keys: {len(unexpected_keys)}")

    return model, optimal_thresholds, epoch, ckpt_pooling_params


def get_operating_threshold(
    pooling_strategy: str,
    override_threshold: Optional[float],
    ckpt_thresholds: Dict[str, float]
) -> float:
    """Determines the operating threshold based on the pooling strategy and override settings."""
    # Determine Operating Decision Threshold
    if override_threshold is not None:
        operating_threshold = override_threshold
    elif pooling_strategy in ckpt_thresholds:
        operating_threshold = ckpt_thresholds.get(pooling_strategy)
    else:
        operating_threshold = 0.5
    return operating_threshold


def resolve_pooling_config(
    pooling_strategy: Optional[str],
    top_percentile: Optional[float],
    t_window: Optional[float],
    ckpt_pooling_params: Dict[str, Union[str, float]]
) -> Tuple[str, float, float]:
    """
    Resolves the effective (pooling_strategy, top_percentile, t_window),
    layering in priority order: explicit CLI flags (non-None) > the pooling
    config saved in the checkpoint at training time > hardcoded fallback
    defaults. Mirrors `get_operating_threshold`'s "checkpoint value unless
    explicitly overridden" pattern, applied to the pooling config that
    produced the checkpoint's own calibrated thresholds -- so inference and
    analysis scripts reproduce training-time pooling by default while still
    letting a caller deliberately try a different strategy via the CLI.
    """
    resolved_strategy = (
        pooling_strategy if pooling_strategy is not None
        else ckpt_pooling_params.get("primary_pooling", "p85_score")
    )
    resolved_top_percentile = (
        top_percentile if top_percentile is not None
        else ckpt_pooling_params.get("top_percentile", 0.10)
    )
    resolved_t_window = (
        t_window if t_window is not None
        else ckpt_pooling_params.get("t_window", 0.60)
    )
    return resolved_strategy, resolved_top_percentile, resolved_t_window


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "macro_f1"
) -> Tuple[float, float]:
    """
    Sweeps decision threshold values from 0.01 to 0.99 to find the threshold 
    that maximizes the specified subject-level performance metric.
    """
    best_t = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.01, 0.99, 99)

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        elif metric == "balanced_accuracy":
            sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
            spec = recall_score(y_true, preds, pos_label=0, zero_division=0)
            score = (sens + spec) / 2.0
        else:
            score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t), float(best_score)


def is_checkpoint_improvement(
    new_f1: float, new_auc: float, best_f1: float, best_auc: float, eps: float = 1e-6
) -> bool:
    """
    Checkpoint-selection criterion: strict Pareto improvement over BOTH subject-level macro F1 and
    AUC -- neither metric may regress, and at least one must strictly improve.

    F1 here is computed at its own per-epoch optimal threshold (find_optimal_threshold() sweeps 99
    thresholds and reports the max), so it only tracks ONE point on the ROC curve -- on a small
    validation cohort (this pipeline's is typically ~35-40 subjects), that's a genuinely high-variance
    statistic: a single subject crossing the decision boundary can swing it, and the fresh 99-way
    threshold sweep every epoch is actively hunting for whatever spike exists in that epoch's noise.
    AUC integrates over the entire ranking rather than depending on exactly where one boundary lands,
    so it's comparatively stable. An F1-only "> best" check (what this replaces) has two failure
    modes: it never saves during a stretch where F1 sits at its own plateau while AUC keeps
    genuinely improving (the original bug), AND it happily accepts an F1 uptick that came at a
    meaningful AUC cost -- plausibly noise-chasing a threshold-sweep spike rather than a real
    improvement. Requiring non-regression on the OTHER metric before crediting an improvement on
    either one guards against both.

    `eps` guards the "hasn't regressed" side against float noise -- without it, a metric that's
    numerically 1e-9 below its prior best (semantically tied) would wrongly veto a real improvement
    in the other one.
    """
    f1_improved = new_f1 > best_f1
    f1_not_worse = new_f1 >= best_f1 - eps
    auc_improved = new_auc > best_auc
    auc_not_worse = new_auc >= best_auc - eps

    return (f1_improved and auc_not_worse) or (auc_improved and f1_not_worse)


def seed_everything(seed: int = 42) -> None:
    """Ensures end-to-end reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


