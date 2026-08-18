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
from scipy.signal import butter, sosfiltfilt
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

from cbramod_stats import spearman_corr


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
        # (p08a_extract_features.py) writes them unconditionally. Fail loudly
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


def flatten_cached_feature_dataset(
    dataset: "CachedFeatureSubjectDataset"
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flattens a (subject-grouped) CachedFeatureSubjectDataset back out to window-level arrays,
    restricted to whatever subject filter the dataset was constructed with (dataset.unique_subjects).

    CachedFeatureSubjectDataset's own __getitem__ groups all of one subject's windows together (one
    item per subject, for patient-level inference) -- the wrong shape for a flat, shuffled window-
    level training batch, or for a window-level forward pass that gets pooled by subject afterward
    (p08b's validation loop). This is the bridge: reuse the class's file-loading and subject-filter-
    parsing (a single pt_path + filter_subject covers train/val/test/any-CV-fold's subject subset,
    all against the SAME master cache, no separate per-split extraction or temporary per-fold cache
    files needed), then flatten back to (feats, labels, subject_ids, stages, indices) at the window
    level for whatever flat batching the caller actually needs.
    """
    mask = np.isin(dataset.subject_ids, dataset.unique_subjects)
    return (
        dataset.feats[mask],
        dataset.labels[mask],
        dataset.subject_ids[mask],
        dataset.stages[mask],
        dataset.indices[mask],
    )


def load_subject_ids(manifest_csv: str) -> List[str]:
    """Reads just the subject_id column of a manifest CSV, raising a clear error if the column is
    missing rather than a KeyError deep inside pandas. Shared by the whole p08b/p13-p21 family --
    this used to be two near-identical functions (one with the validation, one without) split across
    p08b/p20 vs. everything else, until it became clear there was no real reason not to always
    include the clearer error message; merged into one."""
    df = pd.read_csv(manifest_csv)
    if "subject_id" not in df.columns:
        raise ValueError(f"{manifest_csv} has no 'subject_id' column -- is this a p03 split manifest?")
    return df["subject_id"].astype(str).tolist()


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


class GatedAttentionMIL(nn.Module):
    """
    Attention-based deep MIL with a gating mechanism (Ilse, Tomczak & Welling, 2018) -- Option B's
    architecture (p16_gated_attention_embedding_mil.py and friends p17-p19). Unlike Option A's
    single-nonlinearity gate (a plain LayerNorm -> Linear -> Tanh -> Dropout -> Linear MLP), the
    classic gated-attention formulation combines a tanh branch and a sigmoid branch elementwise before
    the final scalar score -- the sigmoid branch acts as a learned gate controlling how much of the
    tanh branch's (potentially non-monotonic) representation contributes, which the original paper
    found more expressive than a tanh-only gate for exactly this kind of "which instances matter" task.

    No fixed bag size anywhere (same invariant as Option A): V/U/w are applied per-window with shared
    weights, and softmax normalizes over whatever bag length is passed at call time.

    forward() returns (logits, attn_weights) for ONE subject's bag -- logits are raw (pre-softmax)
    class scores from the pooled representation, attn_weights are exposed for the same kind of
    interpretability analysis p14 did for Option A.
    """
    def __init__(
        self, num_patches: int = 30, emb_dim: int = 200, attn_hidden_dim: int = 32,
        head_hidden_dim: int = 64, dropout: float = 0.3, num_classes: int = 2, head_type: str = "mlp",
    ):
        super().__init__()
        in_features = num_patches * emb_dim
        self.norm = nn.LayerNorm(in_features)
        self.V = nn.Linear(in_features, attn_hidden_dim)
        self.U = nn.Linear(in_features, attn_hidden_dim)
        self.w = nn.Linear(attn_hidden_dim, 1)
        self.gate_dropout = nn.Dropout(dropout)

        # head_type="linear" isn't just fewer parameters than "mlp" -- it's a qualitatively different
        # interpretability position. Because pooled = sum(attn_weight_i * embedding_i) and a LINEAR
        # head distributes over that sum, logit = W . pooled + b = sum(attn_weight_i * (W . embedding_i))
        # + b -- the final decision decomposes EXACTLY into a per-window "evidence" term (W . embedding_i)
        # weighted by attn_weight_i, the same clean additive structure Option A had
        # (attn_weight_i * window_prob_i), enabling an exact per-window attribution (correlate
        # W . embedding_i against band power, same as p14 did for window_prob) rather than an
        # approximation. An MLP head breaks this: MLP(sum(a_i * x_i)) != sum(a_i * MLP(x_i)) in
        # general, since nonlinear functions don't commute with a weighted sum -- there is no exact
        # per-window decomposition once the head is nonlinear, only approximate attribution methods
        # (leave-one-out, gradients, etc.).
        if head_type == "linear":
            self.head = nn.Linear(in_features, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_features, head_hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type!r} -- choose 'linear' or 'mlp'.")

    def forward(self, bag_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bag_feats: [n_windows, num_patches, emb_dim] -- one subject's windows, n_windows arbitrary."""
        n_windows = bag_feats.shape[0]
        flat = self.norm(bag_feats.reshape(n_windows, -1))  # [n_windows, in_features]
        gated = torch.tanh(self.V(flat)) * torch.sigmoid(self.U(flat))  # [n_windows, attn_hidden_dim]
        scores = self.w(self.gate_dropout(gated)).squeeze(-1)  # [n_windows]
        attn_weights = torch.softmax(scores, dim=0)  # normalizes over THIS bag's actual size
        pooled = (attn_weights.unsqueeze(-1) * flat).sum(dim=0)  # [in_features] -- weighted sum of embeddings
        logits = self.head(pooled.unsqueeze(0)).squeeze(0)  # [num_classes]
        return logits, attn_weights


class AttentionPoolingHead(nn.Module):
    """
    Option A's attention gate (p13_attention_mil_pooling.py and friends p14/p15) -- a learned
    replacement for compute_pooled_scores(method="p85_score"). Given one subject's bag of window
    embeddings and that same subject's FROZEN window-level probabilities (already computed by a
    probe head this gate never retrains), learns a per-window attention weight and returns the
    attention-weighted sum of the window probabilities as the subject-level score.

    The gate conditions on the full embedding rather than on the probe's scalar output alone: a
    window-level probability is already a heavy compression of a ~num_patches*emb_dim-dimensional
    embedding down to one number, optimized for a different objective (window-level classification).
    Two windows can land at the same probe output (e.g. both near 0.5) for very different reasons --
    one genuinely ambiguous-but-clean, one noisy/borderline-artifactual -- and that distinction is
    already erased by the time a scalar-only gate would see it. Conditioning on the embedding
    instead gives the gate a chance to learn "how much to trust this window" using information the
    probe's own compression discarded; a gate restricted to a 1-dimensional input would only be able
    to learn some monotonic-ish reweighting of the probe's own score, which is uncomfortably close
    to just being another fixed pooling statistic rather than genuinely contextual attention.

    The quantity being POOLED, though, is still the already-validated, already-trained window-level
    probability, not the embedding itself -- this keeps the pooled score directly comparable to
    every prior pooling strategy (p85_score, top_10_mean, ...) and keeps the probe head itself
    completely untouched -- unlike GatedAttentionMIL (Option B), which pools the embedding itself.
    """
    def __init__(self, num_patches: int = 30, emb_dim: int = 200, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        in_features = num_patches * emb_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bag_feats: torch.Tensor, window_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bag_feats:    [n_windows, num_patches, emb_dim] -- ONE subject's windows. n_windows varies
                      call to call; nothing about self.gate's parameters depends on it.
        window_probs: [n_windows] -- frozen probe's P(class=1) per window, already computed
                      upstream with no_grad().

        Returns (subject_prob, attn_weights): subject_prob is a 0-dim tensor, attn_weights is
        [n_windows] (sums to 1) -- exposed for inspection/plotting, same role as the leave-one-out
        contribution and LOO-influence analyses used to characterize p85 pooling.
        """
        n_windows = bag_feats.shape[0]
        flat = bag_feats.reshape(n_windows, -1)          # [n_windows, num_patches * emb_dim]
        scores = self.gate(flat).squeeze(-1)              # [n_windows]
        attn_weights = torch.softmax(scores, dim=0)        # normalizes over THIS bag's actual size
        subject_prob = (attn_weights * window_probs).sum()
        return subject_prob, attn_weights


@torch.no_grad()
def frozen_window_probs(probe: nn.Module, bag_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    """P(class=1) per window, from a frozen probe (build_frozen_probe()'s output), for one subject's
    bag of windows -- shared by Option A's training/eval loop (p13) and its interpretability/
    perturbation follow-ups (p14/p15)."""
    logits = probe(bag_feats.to(device).float())
    return torch.softmax(logits, dim=1)[:, 1]


# spearman_corr lives in cbramod_stats.py (pure numpy, no torch/braindecode) -- imported below rather
# than defined here, so the model-free scripts in this family (p09g/p09j/p22/p23) can use the exact
# same implementation without picking up this module's heavy ML dependencies. Re-exported under this
# name so every other script's existing `from cbramod_common import spearman_corr` keeps working.


def report_reference_correlations(
    df: pd.DataFrame, reference_col: str, feature_cols: List[str], subject_col: str = "subject_id",
) -> None:
    """Prints pooled + within-subject Spearman correlation of `reference_col` against each of
    `feature_cols` -- shared by p14 and p17 (attn_weight/window_evidence vs. band power), which had
    byte-identical copies differing only in print-header wording. A subject needs at least 3 valid
    (non-NaN) rows in a column to contribute to that column's within-subject correlation.

    Distinct from report_probability_correlations() below: this one takes an arbitrary
    `reference_col` (e.g. "attn_weight"), that one hardcodes "probability" as the reference and adds
    a couple of extra conveniences (an empty-df guard, frac(r>0.2)/frac(r<-0.2) reporting) that this
    version doesn't -- kept as two separate functions rather than one over-parameterized one, since
    the two callers' actual needs have diverged this much already."""
    print("\n" + "=" * 88)
    print(f"POOLED CORRELATION ({reference_col} vs. features, all windows/subjects together -- conflates within/between-subject variance)")
    print("=" * 88)
    for col in feature_cols:
        valid = df[col].notna()
        r = spearman_corr(df.loc[valid, reference_col].values, df.loc[valid, col].values)
        print(f"  {reference_col} vs {col:<24}: Spearman r = {r:+.4f}  (n={int(valid.sum())})")

    print("\n" + "=" * 88)
    print(f"WITHIN-SUBJECT CORRELATION ({reference_col} vs. features, summarized across subjects -- the direct test)")
    print("=" * 88)
    for col in feature_cols:
        per_subject_r = []
        for _, g in df.groupby(subject_col):
            valid = g[col].notna()
            if valid.sum() < 3:
                continue
            r = spearman_corr(g.loc[valid, reference_col].values, g.loc[valid, col].values)
            if not np.isnan(r):
                per_subject_r.append(r)
        per_subject_r = np.array(per_subject_r)
        if len(per_subject_r) == 0:
            print(f"  {reference_col} vs {col:<24}: no subjects had enough variance to compute this.")
            continue
        print(
            f"  {reference_col} vs {col:<24}: mean r = {per_subject_r.mean():+.4f}, "
            f"median r = {np.median(per_subject_r):+.4f}, "
            f"frac(r>0.2) = {(per_subject_r > 0.2).mean():.2f}, "
            f"frac(r<-0.2) = {(per_subject_r < -0.2).mean():.2f} (n_subjects={len(per_subject_r)})"
        )


def report_probability_correlations(df: pd.DataFrame, feature_cols: List[str], label: str) -> None:
    """Prints pooled + within-subject Spearman correlation of the model's own "probability" column
    against each of `feature_cols` -- shared by p09f and p09k, which had near-identical copies (p09f's
    was the more complete of the two: an empty-df guard and frac(r>0.2)/frac(r<-0.2) reporting p09k
    lacked, both included here so p09k gains them rather than losing anything). A subject needs at
    least 5 windows AND nonzero variance in a column to contribute to that column's within-subject
    correlation -- a different (stricter) bar than report_reference_correlations()'s n>=3, kept as-is
    rather than silently harmonized, since changing either threshold would change reported numbers."""
    print("\n" + "=" * 88)
    print(f"POOLED CORRELATION -- {label} (all windows, all subjects together -- conflates within/between-subject variance)")
    print("=" * 88)
    if len(df) == 0:
        print("  (no rows in this subset)")
        return
    for col in feature_cols:
        r = spearman_corr(df["probability"].values, df[col].values)
        print(f"  probability vs {col:20s}: Spearman r = {r:+.4f}  (n={len(df)})")

    print("\n" + "-" * 88)
    print(f"WITHIN-SUBJECT CORRELATION -- {label} (summarized across subjects -- the direct test)")
    print("-" * 88)
    for col in feature_cols:
        per_subject_r = []
        for _, group in df.groupby("subject_id"):
            if len(group) >= 5 and group[col].std() > 0:
                r = spearman_corr(group["probability"].values, group[col].values)
                if not np.isnan(r):
                    per_subject_r.append(r)
        per_subject_r = np.array(per_subject_r)
        if len(per_subject_r) == 0:
            print(f"  probability vs {col:20s}: no subjects had enough variance to compute this.")
            continue
        frac_pos = np.mean(per_subject_r > 0.2)
        frac_neg = np.mean(per_subject_r < -0.2)
        print(
            f"  probability vs {col:20s}: mean r = {per_subject_r.mean():+.4f}, "
            f"median r = {np.median(per_subject_r):+.4f}, "
            f"frac(r>0.2) = {frac_pos:.2f}, frac(r<-0.2) = {frac_neg:.2f} "
            f"(n_subjects={len(per_subject_r)})"
        )


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


BAND_DEFS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}


def perturb_window_band_power(
    window_CT: np.ndarray, sfreq: float, low: float, high: float, scale_factor: float, order: int = 4,
    preserve_total_energy: bool = True
) -> np.ndarray:
    """
    Rescales the [low, high] Hz component of every channel by `scale_factor`, leaving everything
    else (and zero-padded/inactive channels) untouched. Applying the SAME scale to every channel
    means the model's own channel-mean pooling step scales its band power by exactly the same
    factor (mean is linear), so this targets a predictable, well-defined quantity rather than an
    arbitrary single-channel edit. Used by p09h (window-level perturbation) and p09i (subject-level,
    all-windows-at-once perturbation) -- previously duplicated verbatim in both, now shared here after
    a fix (vectorizing the channel loop) had to be applied to both copies and briefly drifted out of
    sync between them.

    `preserve_total_energy` (default True) renormalizes each perturbed channel back to its
    ORIGINAL std after the band rescale. Without this, scaling a band that already dominates a
    channel's total power (e.g. delta, often 80%+ of relative power) also substantially changes
    the channel's OVERALL amplitude -- since the model's input is Z-scored to std~=1 per channel
    per window (see p02_slice_eeg_dataset.py), that overall-amplitude shift pushes the perturbed
    signal outside the range every training window was normalized to, confounding "does the model
    care about this band's share of the spectrum" with "does the model react to an unrealistically
    high/low-energy window." For a minor band like sigma (~5% of power) this confound is small
    (~3% amplitude shift at scale=1.5); for a dominant band like delta it can be large (~44% at the
    same scale factor) -- big enough to plausibly explain a result on its own. Renormalizing here
    isolates the intended effect (spectral shape) from this confound (overall energy). At
    scale_factor=1.0 this is an exact no-op (renormalization factor is exactly 1.0) either way.

    Vectorized across all channels via sosfiltfilt's `axis` parameter (one call over the whole [C, T]
    array, not a Python loop calling it once per channel) -- num_windows x len(scale_factors) x 64
    individual filter calls per subject was the dominant cost of both scripts' runtime in practice;
    the GPU forward pass, batched, is comparatively negligible. Filtering an all-zero (zero-padded)
    channel just produces zero output, so it's safe -- and exactly equivalent, validated against the
    old per-channel loop on synthetic data -- to run every channel uniformly rather than skip-check
    each one; the preserve_total_energy step's new_std > 1e-8 guard already leaves those channels at
    exactly zero afterward too.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    band_component = sosfiltfilt(sos, window_CT, axis=-1)
    residual = window_CT - band_component
    new_sig = residual + scale_factor * band_component

    if preserve_total_energy:
        orig_std = window_CT.std(axis=-1, keepdims=True)
        new_std = new_sig.std(axis=-1, keepdims=True)
        safe_new_std = np.where(new_std > 1e-8, new_std, 1.0)
        rescale = np.where(new_std > 1e-8, orig_std / safe_new_std, 1.0)
        new_sig = new_sig * rescale

    return new_sig.astype(window_CT.dtype)


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

    def train(self, *args, **kwargs) -> dict:
        """
        Placeholder -- concrete subclasses define their own train() signature, since their actual
        data-loading needs genuinely differ (EndToEndTrainer takes fixed train/val manifest paths;
        ProbeTrainer takes a single master cache path and resolves train/val subject subsets from it
        internally). Deliberately not pinned to one concrete signature here: nothing calls train()
        polymorphically through this base class today, and forcing a shared arg list would be
        artificial rather than a real interface both subclasses actually implement.
        """
        raise NotImplementedError("Subclasses must implement train().")


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


def setup_cache_cli_parser(parser: argparse.ArgumentParser) -> None:
    """Adds --cache-dir/--master-cache-name -- shared by every script that reads from the master
    feature cache built by p08a_extract_features.py (p13/p14/p16/p17/p18/p20/p21), previously
    duplicated verbatim across all of them."""
    cache_group = parser.add_argument_group("Cache")
    cache_group.add_argument("--cache-dir", type=str, required=True, help="Directory containing the master cache")
    cache_group.add_argument("--master-cache-name", type=str, default="cached_master_embeddings.pt")


def add_log_filename_argument(parser: argparse.ArgumentParser, script_file: str) -> None:
    """Adds --log-filename with a default derived from the calling script's own filename (pass
    __file__ from the calling script). A tiny shared helper purely so every script's --log-filename
    has identical help text/behavior -- the one-liner itself wasn't hard to duplicate, but it had
    drifted into a few slightly different phrasings across the p13-p21 family."""
    group = parser.add_argument_group("Logging")
    group.add_argument(
        "--log-filename", type=str, default=Path(script_file).stem + ".log",
        help="Filename for this script's own log output."
    )


def setup_perturbation_cli_parser(
    parser: argparse.ArgumentParser,
    output_csv_default: str,
    max_windows_per_subject_default: Optional[int] = None,
) -> None:
    """Adds the CLI flags shared by every band-power perturbation test in this project
    (p09h/p09i/p15/p19): band choice, scale-factor grid, filter order, total-energy preservation,
    a subsampling cap, and a subject-selection JSON. --output-csv's default and
    --max-windows-per-subject's default vary per script (p09h alone defaults the latter to 40, not
    None), so both are passed in rather than hardcoded. --perturb-fraction (p09i's graded-subset
    dose-response sweep) is intentionally NOT included here -- add it separately where needed."""
    group = parser.add_argument_group("Perturbation Test")
    group.add_argument(
        "--band", type=str, default="sigma", choices=list(BAND_DEFS.keys()),
        help="Which frequency band to perturb."
    )
    group.add_argument(
        "--scale-factors", type=str, default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated grid of scale factors applied to the band's amplitude "
             "(1.0 = unperturbed original)."
    )
    group.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order for band isolation.")
    group.add_argument(
        "--no-preserve-total-energy", dest="preserve_total_energy", action="store_false",
        help="Disable renormalizing each perturbed channel back to its original std after the band "
             "rescale (see perturb_window_band_power()'s docstring). Default: preserve_total_energy=True."
    )
    group.add_argument(
        "--max-windows-per-subject", type=int, default=max_windows_per_subject_default,
        help="Optional cap on windows perturbed/analyzed per subject, for runtime control."
    )
    group.add_argument(
        "--subjects-json", type=str, default=None,
        help="Path to a p09d_subject_confidence_report.py --output-json report. Its subject_ids are "
             "unioned with --subject-id (if also given) to select which subjects to analyze."
    )
    group.add_argument("--output-csv", type=str, default=output_csv_default)


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
    ckpt_group.add_argument(
        "--checkpoint", "--probe-checkpoint", dest="probe_checkpoint", type=str, required=True,
        help="Path to the p08b-trained probe checkpoint (.pt). --probe-checkpoint is accepted as an "
             "alias for the same flag, matching the naming used by p13/p14/p20/p21."
    )

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


def extract_ckpt_metadata(checkpoint: dict) -> Tuple[dict, Union[int, str], Dict[str, Union[str, float]]]:
    """
    Pulls (optimal_thresholds, epoch, pooling_params) off a raw checkpoint dict -- the same
    extraction load_model_checkpoint() has always done internally, now also usable directly by
    callers of build_frozen_e2e_classifier()/build_gated_attention_model() (p09i/p15), which return
    the raw checkpoint dict instead of this pre-extracted tuple. Only keys actually present are
    included in pooling_params, so older checkpoints saved before this field existed degrade
    gracefully to an empty dict.
    """
    optimal_thresholds = checkpoint.get("optimal_thresholds", {}) if isinstance(checkpoint, dict) else {}
    epoch = checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"
    ckpt_pooling_params = {
        key: checkpoint[key]
        for key in ("primary_pooling", "top_percentile", "t_window")
        if isinstance(checkpoint, dict) and key in checkpoint
    }
    return optimal_thresholds, epoch, ckpt_pooling_params


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

    optimal_thresholds, epoch, ckpt_pooling_params = extract_ckpt_metadata(checkpoint)

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


def infer_probe_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """
    Infers head_type (and, for MLP, hidden_dim) directly from a probe checkpoint's own state_dict
    keys/shapes, rather than trusting --head-type/--head-dim to happen to match whatever the
    checkpoint was actually trained with. A probe checkpoint only stores weights, not architecture --
    --head-type's default is "mlp" (setup_common_cli_parser's shared default across every script that
    uses it), but this investigation has mostly trained and used the LINEAR head, so relying on the
    flag alone is a real, easy-to-hit footgun: forgetting --head-type=linear would try to
    load_state_dict() a linear checkpoint into an MLPProbeHead, which raises a size-mismatch/missing-key
    RuntimeError -- not silent corruption, but also not an obviously-actionable message pointing at
    the real cause. Determining architecture from the file itself removes the whole failure mode.

    LinearProbeHead.head is Sequential(Rearrange, LayerNorm, Linear) -> keys head.1.*, head.2.*.
    MLPProbeHead.head is Sequential(Rearrange, LayerNorm, Linear, ELU, Dropout, Linear) -> keys
    head.1.*, head.2.*, head.5.* (ELU/Dropout have no parameters, so no head.3.*/head.4.* keys exist).
    """
    if "head.5.weight" in state_dict:
        return {"head_type": "mlp", "hidden_dim": int(state_dict["head.2.weight"].shape[0])}
    elif "head.2.weight" in state_dict:
        return {"head_type": "linear"}
    raise KeyError(
        f"Could not recognize probe checkpoint architecture from its state_dict keys "
        f"({sorted(state_dict.keys())}) -- expected either LinearProbeHead's keys ('head.1.*', "
        f"'head.2.*') or MLPProbeHead's keys ('head.1.*', 'head.2.*', 'head.5.*')."
    )


def resolve_probe_architecture(checkpoint_path, config: argparse.Namespace, logger) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], dict]:
    """
    Resolves a p08b/p20-trained probe checkpoint's ACTUAL head architecture (head_type, and
    hidden_dim for MLP), NEVER trusting --head-type/--head-dim blindly. Priority order:
      1. Explicit metadata saved IN the checkpoint itself (p08b/p20 write "head_type"/"head_dim" at
         save time) -- the single source of truth going forward, no guessing involved.
      2. For checkpoints that predate that fix, fall back to inferring architecture from the
         state_dict's own key/shape pattern (infer_probe_architecture()) -- still derived from the
         file itself, just less direct than explicit metadata.
    Either way, --head-type/--head-dim are only ever used as a cross-check (mismatch logged loudly),
    never as the actual source of the reconstructed architecture. Shared by both consumers of a
    probe checkpoint: build_frozen_probe() (bare head, for cached-embedding scripts) and
    build_frozen_e2e_classifier() (full backbone+head, for raw-waveform scripts).

    Returns (resolved_architecture, head_state_dict, raw_checkpoint_dict) -- callers construct the
    module that fits their own use case and load the state dict into the right place themselves.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if "head_type" in ckpt:
        resolved = {"head_type": ckpt["head_type"]}
        if ckpt["head_type"] == "mlp":
            resolved["hidden_dim"] = ckpt["head_dim"]
    else:
        logger.warning(
            f"Probe checkpoint ({checkpoint_path}) has no explicit head_type metadata -- it predates "
            f"that being saved. Falling back to inferring architecture from the state_dict's own "
            f"key/shape pattern instead of guessing from --head-type."
        )
        resolved = infer_probe_architecture(state_dict)

    if resolved["head_type"] != config.head_type or (
        resolved["head_type"] == "mlp" and resolved.get("hidden_dim") != config.head_dim
    ):
        logger.warning(
            f"--head-type/--head-dim ({config.head_type}/{config.head_dim}) do NOT match what the "
            f"probe checkpoint ({checkpoint_path}) actually is ({resolved}) -- using the checkpoint's "
            f"own architecture (this is what actually gets built), not the CLI flags. If this is "
            f"unexpected, double check the checkpoint path points at the file you think it does."
        )

    return resolved, state_dict, ckpt


def build_frozen_probe(config: argparse.Namespace, device: torch.device, logger) -> nn.Module:
    """
    Reconstructs the bare probe head architecture and loads frozen weights from
    config.probe_checkpoint -- for scripts operating on already-extracted cached embeddings
    (p13/p14/p20/p21), which only ever need the head, not the CBraMod backbone. See
    resolve_probe_architecture() for the metadata-first resolution this relies on.
    """
    resolved, state_dict, _ckpt = resolve_probe_architecture(config.probe_checkpoint, config, logger)

    if resolved["head_type"] == "linear":
        probe = LinearProbeHead(num_patches=config.num_patches, emb_dim=config.cbra_dim, num_classes=config.num_classes)
    else:
        probe = MLPProbeHead(
            num_patches=config.num_patches, emb_dim=config.cbra_dim,
            hidden_dim=resolved["hidden_dim"], num_classes=config.num_classes, dropout=config.dropout,
        )

    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def build_frozen_e2e_classifier(config: argparse.Namespace, device: torch.device, logger) -> Tuple[nn.Module, dict]:
    """
    Same metadata-first architecture resolution as build_frozen_probe(), but constructs the full
    CBraModE2EClassifier (fresh pretrained backbone + the checkpoint's own head) instead of a bare
    head -- for scripts that need raw-waveform inference (p09f/p09h/p09i/p15), since they must run
    the backbone forward pass themselves on (possibly perturbed) raw signal rather than reading
    pre-extracted embeddings from the master cache.

    Replaces the previous pattern in these scripts (construct a CBraModE2EClassifier purely from
    --head-type/--head-dim, then load_model_checkpoint() the head weights in afterward, discovering
    any mismatch only via a shape-mismatch RuntimeError) with the same metadata-first, loud-warning
    discipline p13/p14/p20/p21 already had for the bare-head case.

    Returns (model, raw_checkpoint_dict) -- the checkpoint dict is returned too so callers can read
    optimal_thresholds/primary_pooling/etc. off it themselves, same information load_model_checkpoint()
    used to hand back as a tuple.
    """
    resolved, state_dict, ckpt = resolve_probe_architecture(config.probe_checkpoint, config, logger)

    model = CBraModE2EClassifier(
        num_channels=config.num_channels, sfreq=config.sfreq, num_patches=config.num_patches,
        emb_dim=config.cbra_dim, hidden_dim=resolved.get("hidden_dim", config.head_dim),
        num_classes=config.num_classes, dropout=config.dropout, head_type=resolved["head_type"],
    )
    model.head.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def build_gated_attention_model(
    config: argparse.Namespace, device: torch.device, logger, require_head_type: Optional[str] = None,
) -> Tuple[nn.Module, dict]:
    """
    Reconstructs a GatedAttentionMIL (Option B) architecture and loads frozen weights from
    config.model_checkpoint, reading the checkpoint's own saved metadata
    (attn_hidden_dim/head_hidden_dim/num_patches/cbra_dim/num_classes/head_type) with a LOUD warning
    on any mismatch against the corresponding CLI flags -- replacing the silent
    `ckpt.get(key, args.xxx)` pattern previously duplicated across p17/p18/p19, which accepted a
    mismatch with no warning at all.

    require_head_type: if given (e.g. "linear"), raises with an explanation if the checkpoint's own
    head_type doesn't match -- p17/p18 both need this since they rely on the exact per-window linear
    decomposition, which only holds for a linear head.

    Returns (model, raw_checkpoint_dict) -- callers read optimal_threshold/epoch/etc. off the dict.
    """
    ckpt = torch.load(config.model_checkpoint, map_location="cpu", weights_only=True)
    head_type = ckpt.get("head_type", "mlp")

    if require_head_type is not None and head_type != require_head_type:
        raise ValueError(
            f"--model-checkpoint has head_type={head_type!r}, not {require_head_type!r}. This script "
            f"relies on the exact linear-head decomposition (head(pooled) == "
            f"sum(attn_weight_i * head(flat_i))), which only holds for a linear head -- an MLP head "
            f"does not commute with the weighted sum. Train (or load) a --head-type linear p16 "
            f"checkpoint instead."
        )

    resolved = {
        "attn_hidden_dim": ckpt.get("attn_hidden_dim", config.attn_hidden_dim),
        "head_hidden_dim": ckpt.get("head_hidden_dim", config.head_hidden_dim),
        "num_patches": ckpt.get("num_patches", config.num_patches),
        "cbra_dim": ckpt.get("cbra_dim", config.cbra_dim),
        "num_classes": ckpt.get("num_classes", config.num_classes),
        "head_type": head_type,
    }
    for key in ("attn_hidden_dim", "head_hidden_dim"):
        cli_value = getattr(config, key, None)
        if cli_value is not None and resolved[key] != cli_value:
            logger.warning(
                f"--{key.replace('_', '-')} ({cli_value}) does NOT match what --model-checkpoint "
                f"actually is ({resolved[key]}) -- using the checkpoint's own value (this is what "
                f"actually gets built), not the CLI flag."
            )

    model = GatedAttentionMIL(
        num_patches=resolved["num_patches"], emb_dim=resolved["cbra_dim"],
        attn_hidden_dim=resolved["attn_hidden_dim"], head_hidden_dim=resolved["head_hidden_dim"],
        dropout=0.0, num_classes=resolved["num_classes"], head_type=resolved["head_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"Loaded Option B ({resolved['head_type']} head) model from {config.model_checkpoint} "
        f"(epoch {ckpt.get('epoch', '?')})"
    )
    return model, ckpt


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
    new_f1: float, new_auc: float, best_f1: float, best_auc: float,
    eps: float = 1e-6, min_large_gain: float = 0.02, max_small_dip: float = 0.01
) -> bool:
    """
    Checkpoint-selection criterion: strict Pareto improvement over BOTH subject-level macro F1 and
    AUC (neither metric may regress, at least one must strictly improve) -- OR a large gain in one
    metric alongside only a small dip in the other.

    F1 here is computed at its own per-epoch optimal threshold (find_optimal_threshold() sweeps 99
    thresholds and reports the max), so it only tracks ONE point on the ROC curve -- on a small
    validation cohort (this pipeline's is typically ~35-40 subjects), that's a genuinely high-variance
    statistic: a single subject crossing the decision boundary can swing it, and the fresh 99-way
    threshold sweep every epoch is actively hunting for whatever spike exists in that epoch's noise.
    AUC integrates over the entire ranking rather than depending on exactly where one boundary lands,
    so it's comparatively stable. An F1-only "> best" check (what this originally replaced) has two
    failure modes: it never saves during a stretch where F1 sits at its own plateau while AUC keeps
    genuinely improving, AND it happily accepts an F1 uptick that came at a meaningful AUC cost --
    plausibly noise-chasing a threshold-sweep spike rather than a real improvement. Requiring
    non-regression on the OTHER metric before crediting an improvement on either one guards against
    both (e.g. a real epoch: F1 +0.0005 / AUC -0.0119 is correctly rejected -- the "gain" is smaller
    than one subject's worth of macro-F1 movement, the "dip" is an order of magnitude larger).

    The pure strict-Pareto rule is arguably too conservative in the OTHER direction, though: it would
    also reject a case where one metric jumps a lot and the other only dips a little -- a plausible
    real improvement, not noise. The min_large_gain/max_small_dip clause recovers that case WITHOUT
    reopening the noise-chasing problem above, because it requires the gain to clear an absolute
    "clearly not noise" floor (default 0.02) AND the dip to stay under a "clearly minor" ceiling
    (default 0.01) -- a pure gain/dip RATIO alone wouldn't do this safely, since e.g. a 0.001 gain
    against a 0.0002 dip is a 5x ratio but both numbers are still noise-level; requiring the gain
    itself to be large in absolute terms avoids that.

    `eps` guards the strict-Pareto "hasn't regressed" side against float noise -- without it, a
    metric that's numerically 1e-9 below its prior best (semantically tied) would wrongly veto a real
    improvement in the other one.
    """
    f1_improved = new_f1 > best_f1
    f1_not_worse = new_f1 >= best_f1 - eps
    auc_improved = new_auc > best_auc
    auc_not_worse = new_auc >= best_auc - eps

    strict_pareto = (f1_improved and auc_not_worse) or (auc_improved and f1_not_worse)

    # `eps` tolerance applied on both sides here too, same rationale as the strict-Pareto check above --
    # without it, a gain/dip landing within float noise of the threshold could flicker based on
    # floating-point representation alone rather than a real difference.
    f1_large_gain_small_dip = (new_f1 - best_f1) >= min_large_gain - eps and (new_auc - best_auc) >= -max_small_dip - eps
    auc_large_gain_small_dip = (new_auc - best_auc) >= min_large_gain - eps and (new_f1 - best_f1) >= -max_small_dip - eps

    return strict_pareto or f1_large_gain_small_dip or auc_large_gain_small_dip


def seed_everything(seed: int = 42) -> None:
    """Ensures end-to-end reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


