"""
p10_feature_attribution.py

Computes and visualizes Integrated Gradients (IG) feature attribution maps for
EEG windows flagged as diagnostically interesting by
`p09c_clinical_subject_diagnostics.py` (its 4-tier "priority window" JSON
exports), or for an arbitrary user-specified list of raw window indices.

--------------------------------------------------------------------------
What is "feature attribution" and how is it computed here?
--------------------------------------------------------------------------
Feature attribution answers: "for this one prediction, on this one input, how
much did each input feature contribute to pushing the model's output toward
the predicted class?" It explains a single forward pass, not the model's
weights in general — a different window, even from the same subject, gets its
own independent attribution map.

We use Integrated Gradients (IG; Sundararajan et al., 2017, "Axiomatic
Attribution for Deep Networks") via Captum's `IntegratedGradients`. IG:

  1. Picks a "baseline" input representing the *absence* of signal — here an
     all-zeros EEG window (`torch.zeros_like(input_tensor)`), i.e. "no
     electrical activity".
  2. Constructs a straight-line path in input space from that baseline to the
     real input window, sampled at `n_steps` evenly-spaced interpolation
     points (alpha = 0 -> 1).
  3. At each interpolated point, computes the gradient of the target class's
     output logit with respect to the input.
  4. Averages those gradients along the path and multiplies by
     (input - baseline), approximating the path integral

        IG_i = (x_i - baseline_i) * integral_0^1 dF(baseline + alpha*(x-baseline))/dx_i d(alpha)

     via a Riemann sum over the `n_steps` samples.

  The result is one attribution score per input element (same shape as the
  input EEG tensor, [Channels, Time_Samples]): positive values mean that
  sample pushed the prediction toward the target class, negative values mean
  it pushed away from it, and magnitude reflects how much. Captum also
  returns a convergence delta — how well the completeness axiom
  (sum(attributions) ~= F(input) - F(baseline)) holds for this sample; a large
  delta means `--n-steps` should be increased.

We then aggregate the raw signal-level map into two coarser granularities:
  - Patch-level: CBraMod tokenizes the signal into 1-second patches (patch
    length in samples == `sfreq`, since a window's duration in seconds is
    exactly `num_patches` under this fixed 1s/patch design — see
    `patch_size_samples` below). We L2-norm the per-sample attributions
    within each patch to get one importance score per (channel, patch) cell,
    matching CBraMod's own tokenization grid.
  - Channel-level: L2-norm across the entire time axis, collapsing to one
    score per EEG channel (spatial importance, independent of timing).
--------------------------------------------------------------------------
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from cbramod_common import CBraModE2EClassifier, load_model_checkpoint, setup_inference_cli_parser
from cbramod_utils import seed_everything
from p09c_clinical_subject_diagnostics import SubjectEEGInspector


# -----------------------------------------------------------------------------
# Attribution computation
# -----------------------------------------------------------------------------

def compute_multi_granularity_attributions(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    patch_size_samples: int = 200,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda")
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Integrated Gradients across three structural granularities:
    1. Signal level: [Channels, Time_Samples] (Full resolution)
    2. Patch level:  [Channels, Num_Patches]  (Aggregated per CBraMod patch)
    3. Channel level:[Channels]              (Spatial magnitude)

    See the module docstring for a full explanation of Integrated Gradients
    and how these three granularities are derived from it.
    """
    if not HAS_CAPTUM:
        raise ImportError("Captum library required. Run `pip install captum`.")

    model.eval()
    ig = IntegratedGradients(model)

    input_tensor = input_tensor.to(device)
    baseline = torch.zeros_like(input_tensor).to(device)

    # 1. Compute Raw Signal-Level Attributions [1, C, T]
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    print(f"  [Captum] IG Convergence Delta: {delta.item():.6f}")

    signal_attr = attributions.squeeze(0).cpu().detach().numpy()  # [Channels, Time_Samples]
    num_channels, time_samples = signal_attr.shape

    # 2. Compute Patch-Level Attributions [Channels, Num_Patches]
    # Reshape time dimension into discrete temporal patches
    num_patches = time_samples // patch_size_samples
    truncated_samples = num_patches * patch_size_samples

    # Reshape signal attributions to [Channels, Num_Patches, Patch_Size] and compute L2 norm per patch
    reshaped_attr = signal_attr[:, :truncated_samples].reshape(num_channels, num_patches, patch_size_samples)
    patch_attr = np.linalg.norm(reshaped_attr, axis=2)  # [Channels, Num_Patches]

    # 3. Compute Spatial Channel-Level Attributions [Channels]
    channel_attr = np.linalg.norm(signal_attr, axis=1)  # [Channels]

    return signal_attr, patch_attr, channel_attr


# -----------------------------------------------------------------------------
# Channel concentration: is this a focal or a distributed attribution?
# -----------------------------------------------------------------------------

def _gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a nonnegative array: 0.0 means attribution is spread
    perfectly evenly across channels, 1.0 means it is concentrated entirely
    in a single channel. Used as a cheap, threshold-able stand-in for
    "is there a clear winning channel, or several channels driving this
    together?" without requiring a human to eyeball the bar chart.
    """
    x = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def compute_channel_concentration(
    channel_attr: np.ndarray,
    top_k: int = 3
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Summarizes `channel_attr` [Channels] into:
      - gini: concentration coefficient (see `_gini_coefficient`)
      - top_k_idx: indices of the top_k channels by attribution magnitude,
        descending
      - top_k_scores: their corresponding attribution scores
    """
    top_k = min(top_k, len(channel_attr))
    order = np.argsort(channel_attr)[::-1]
    top_k_idx = order[:top_k]
    return _gini_coefficient(channel_attr), top_k_idx, channel_attr[top_k_idx]


def plot_attribution_dashboard(
    raw_eeg: np.ndarray,
    signal_attr: np.ndarray,
    patch_attr: np.ndarray,
    channel_attr: np.ndarray,
    output_path: Path,
    target_channel_idx: Optional[int] = None,
    sfreq: float = 200.0,
    concentration_threshold: float = 0.5,
    top_k_channels: int = 3,
    gini: Optional[float] = None,
    top_k_idx: Optional[np.ndarray] = None
):
    """
    Generates a visual diagnostic dashboard comparing raw EEG signals,
    signal-level saliency, and CBraMod patch grid importance.

    Panel 1 adapts to whether attribution is focal or distributed: when the
    channel-attribution Gini coefficient is at or above
    `concentration_threshold`, a single dominant channel drives the
    prediction and is plotted alone (as before). Below that threshold, no
    single channel "wins" clearly, so the top `top_k_channels` channels are
    overlaid together instead of picking one arbitrarily.

    `gini`/`top_k_idx` can be passed in pre-computed (e.g. from `main()`, which
    already needs them for logging/export) to avoid recomputing; if omitted
    they're derived here from `channel_attr`.
    """
    time_axis = np.arange(raw_eeg.shape[1]) / sfreq
    num_channels, num_patches = patch_attr.shape

    if gini is None or top_k_idx is None:
        gini, top_k_idx, _ = compute_channel_concentration(channel_attr, top_k=top_k_channels)
    is_focal = gini >= concentration_threshold

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1, 1.2]})

    # --- Panel 1: Raw EEG Trace with Overlay Signal Attribution ---
    if target_channel_idx is not None:
        # Explicit user override always wins, regardless of concentration.
        plot_channels = [target_channel_idx]
    elif is_focal:
        plot_channels = [int(top_k_idx[0])]
    else:
        plot_channels = [int(c) for c in top_k_idx]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(plot_channels), 2)))
    for i, ch in enumerate(plot_channels):
        eeg_signal = raw_eeg[ch]
        attr_signal = signal_attr[ch]
        color = colors[i]

        axes[0].plot(time_axis, eeg_signal, color=color, alpha=0.7, label=f'Raw EEG (Ch {ch})')
        pos_attr = np.maximum(0, attr_signal)
        if pos_attr.max() > 0:
            pos_attr_norm = pos_attr / pos_attr.max() * np.abs(eeg_signal).max()
            axes[0].fill_between(time_axis, 0, pos_attr_norm, color=color, alpha=0.25)

    axes[0].set_ylabel('Amplitude (µV)')
    if target_channel_idx is not None:
        title = f'Signal-Level Attribution (Channel {target_channel_idx}, User-Selected)'
    elif is_focal:
        title = f'Signal-Level Attribution (Channel {plot_channels[0]}, Highest Attribution, Gini={gini:.2f})'
    else:
        title = f'Signal-Level Attribution (Top {len(plot_channels)} Channels, No Clear Winner, Gini={gini:.2f})'
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # --- Panel 2: CBraMod Spatial-Temporal Patch Heatmap ---
    im = axes[1].imshow(
        patch_attr,
        aspect='auto',
        cmap='magma',
        origin='lower',
        extent=[0, time_axis[-1], 0, num_channels]
    )
    axes[1].set_ylabel('EEG Channel Index')
    axes[1].set_title('CBraMod Token Patch Importance Heatmap (Channels x Time Patches)')
    fig.colorbar(im, ax=axes[1], orientation='vertical', label='Patch Importance (L2 Norm)')

    # --- Panel 3: Global Channel Importance ---
    axes[2].bar(range(num_channels), channel_attr, color='navy', alpha=0.7)
    axes[2].set_xlabel('EEG Channel Index')
    axes[2].set_ylabel('Attribution L2 Norm')
    axes[2].set_title('Global Channel Importance Profile')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-granularity attribution plot saved to: {output_path}")


# -----------------------------------------------------------------------------
# Priority Window Resolution
# -----------------------------------------------------------------------------

def _tier_slug(tier_name: str) -> str:
    """Turns a tier name like 'Tier 1: Top Drivers' into 'tier_1_top_drivers'."""
    return tier_name.lower().replace(":", "").replace(" ", "_")


def _resolve_tier_filter(tiers_arg: Optional[str]) -> Optional[List[str]]:
    """
    Maps a user-supplied, possibly-abbreviated `--tiers` value onto the
    canonical tier names in `SubjectEEGInspector.TIER_STYLES` — the single
    source of truth for tier names, shared with p09c so the two can't drift.
    Returns None (no filtering, i.e. all tiers) if `tiers_arg` is falsy.
    """
    if not tiers_arg:
        return None
    all_tiers = list(SubjectEEGInspector.TIER_STYLES.keys())
    resolved: List[str] = []
    for requested in (t.strip() for t in tiers_arg.split(",") if t.strip()):
        matches = [t for t in all_tiers if requested.lower() in t.lower()]
        if not matches:
            raise ValueError(f"--tiers value '{requested}' does not match any known tier: {all_tiers}")
        resolved.extend(matches)
    return resolved


def _load_windows_from_payload(payload: Dict, tier_filter: Optional[List[str]]) -> List[Dict]:
    """Flattens a p09c `*_priority_windows.json` payload into a list of window tasks."""
    tasks = []
    for tier_name, windows in payload.get("priority_tiers", {}).items():
        if tier_filter is not None and tier_name not in tier_filter:
            continue
        for w in windows:
            tasks.append({
                "raw_epoch_index": w["raw_epoch_index"],
                "tier": tier_name,
                "probability": w.get("probability"),
                "target_class": payload.get("prediction"),
            })
    return tasks


def resolve_priority_windows(
    source: str,
    subject_id: Optional[str],
    tiers_arg: Optional[str]
) -> Dict[str, List[Dict]]:
    """
    Resolves `--priority-windows` into {subject_id: [window_task, ...]}.

    `source` accepts three forms:
      - Path to a single `<subject>_priority_windows.json` file written by
        p09c_clinical_subject_diagnostics.py.
      - Path to a directory containing multiple such JSON files (one per
        subject); optionally narrowed with `--subject-id`.
      - A comma-separated list of raw window indices (e.g. "12,45,78"),
        bypassing p09c entirely — requires `--subject-id` to name exactly one
        subject, since there is no tier/probability metadata to draw from.
    """
    tier_filter = _resolve_tier_filter(tiers_arg)
    path = Path(source)
    subject_filter = {s.strip() for s in subject_id.split(",")} if subject_id else None

    windows_by_subject: Dict[str, List[Dict]] = {}

    if path.is_dir():
        json_paths = sorted(path.glob("*_priority_windows.json"))
        if not json_paths:
            raise ValueError(f"No '*_priority_windows.json' files found in directory: {path}")
        for json_path in json_paths:
            with open(json_path) as f:
                payload = json.load(f)
            subj = str(payload["subject_id"])
            if subject_filter and subj not in subject_filter:
                continue
            windows_by_subject[subj] = _load_windows_from_payload(payload, tier_filter)

    elif path.is_file():
        with open(path) as f:
            payload = json.load(f)
        subj = str(payload["subject_id"])
        windows_by_subject[subj] = _load_windows_from_payload(payload, tier_filter)

    elif all(part.strip().lstrip("-").isdigit() for part in source.split(",") if part.strip()):
        if not subject_filter or len(subject_filter) != 1:
            raise ValueError(
                "--priority-windows was given as a raw index list; --subject-id must name exactly one subject."
            )
        subj = next(iter(subject_filter))
        windows_by_subject[subj] = [
            {"raw_epoch_index": int(part), "tier": "Manual", "probability": None, "target_class": None}
            for part in source.split(",") if part.strip()
        ]

    else:
        raise ValueError(
            f"--priority-windows value '{source}' is neither an existing JSON file/directory "
            "nor a comma-separated list of integer window indices."
        )

    return windows_by_subject


def resolve_subject_npy_path(manifest_csv: Path, data_dir: Optional[Path], subject_id: str) -> Path:
    """Looks up a subject's raw window .npy path from the inference manifest CSV (same schema PANSubjectEEGDataset reads)."""
    df = pd.read_csv(manifest_csv)
    df["subject_id"] = df["subject_id"].astype(str)
    match = df[df["subject_id"] == str(subject_id)]
    if match.empty:
        raise ValueError(f"Subject '{subject_id}' not found in manifest: {manifest_csv}")
    raw_npy_path = Path(match.iloc[0]["npy_path"])
    if data_dir and not raw_npy_path.is_absolute():
        return data_dir / raw_npy_path
    return raw_npy_path


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

def parse_cli_args() -> argparse.Namespace:
    parser = setup_inference_cli_parser(description="CBraMod Multi-Granularity Feature Attribution")

    attr_group = parser.add_argument_group("Feature Attribution")
    attr_group.add_argument(
        "--priority-windows", type=str, required=True,
        help=(
            "Path to a p09c '<subject>_priority_windows.json' file, a directory of such files, "
            "or a comma-separated list of raw window indices (requires --subject-id to name one subject)."
        )
    )
    attr_group.add_argument(
        "--tiers", type=str, default=None,
        help="Comma-separated tier filter, substring-matched against p09c tier names (e.g. 'top,spikes'). Default: all tiers."
    )
    attr_group.add_argument("--n-steps", type=int, default=50, help="Integrated Gradients interpolation steps")
    attr_group.add_argument(
        "--target-class", type=int, default=None,
        help="Override the IG target class for every window. Default: each window's own subject-level "
             "prediction from p09c (falls back to 1 if unavailable, e.g. in raw-index-list mode)."
    )
    attr_group.add_argument(
        "--target-channel-idx", type=int, default=None,
        help="EEG channel to plot in Panel 1 of the dashboard. Overrides automatic focal/distributed "
             "channel selection below."
    )
    attr_group.add_argument(
        "--concentration-threshold", type=float, default=0.5,
        help="Gini coefficient (0-1) on channel-level attribution above which a window is treated as "
             "'focal' (one dominant channel, plotted alone). Below this, no single channel clearly wins, "
             "so --top-k-channels channels are overlaid together in Panel 1 instead."
    )
    attr_group.add_argument(
        "--top-k-channels", type=int, default=3,
        help="Number of channels to overlay in Panel 1 when attribution is distributed rather than focal."
    )

    args = parser.parse_args()
    if args.features_pt:
        parser.error(
            "Feature attribution requires raw EEG input (--manifest); --features-pt is not supported, "
            "since Integrated Gradients needs the full CBraModE2EClassifier over the raw waveform."
        )
    return args


def main():
    args = parse_cli_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./attribution_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CBraMod tokenizes each 1-second slice of signal as one patch, so a
    # window's duration in seconds is simply `num_patches`, and each patch
    # spans exactly `sfreq` samples — no separate window-length flag needed.
    patch_size_samples = int(round(args.sfreq))
    window_sec = float(args.num_patches)

    print("Instantiating full CBraModE2EClassifier for raw waveform attribution.")
    model = CBraModE2EClassifier(
        num_channels=args.num_channels,
        sfreq=args.sfreq,
        num_patches=args.num_patches,
        emb_dim=args.cbra_dim,
        hidden_dim=args.head_dim,
        num_classes=args.num_classes,
        head_type=args.head_type
    )
    model, _, _ = load_model_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    windows_by_subject = resolve_priority_windows(args.priority_windows, args.subject_id, args.tiers)
    total_windows = sum(len(tasks) for tasks in windows_by_subject.values())
    print(f"Resolved {total_windows} target window(s) across {len(windows_by_subject)} subject(s).")

    for subject_id, tasks in windows_by_subject.items():
        if not tasks:
            print(f"Skipping {subject_id}: no windows matched the requested tier filter.")
            continue

        npy_path = resolve_subject_npy_path(
            Path(args.manifest), Path(args.data_dir) if args.data_dir else None, subject_id
        )
        subj_data = np.load(npy_path, mmap_mode="r")

        for task in tasks:
            raw_idx = task["raw_epoch_index"]
            if raw_idx >= subj_data.shape[0]:
                print(f"Skipping {subject_id} window {raw_idx}: out of bounds for array with {subj_data.shape[0]} rows.")
                continue

            target_class = args.target_class
            if target_class is None:
                target_class = task["target_class"] if task["target_class"] is not None else 1
            window_sample = np.array(subj_data[raw_idx : raw_idx + 1], dtype=np.float32)  # [1, C, T]
            input_tensor = torch.from_numpy(window_sample)

            tier_tag = _tier_slug(task["tier"])
            print(f"\n=== {subject_id} | window {raw_idx} (@ {raw_idx * window_sec:.0f}s) | "
                  f"{task['tier']} | target_class={target_class} ===")

            signal_attr, patch_attr, channel_attr = compute_multi_granularity_attributions(
                model=model,
                input_tensor=input_tensor,
                target_class=target_class,
                patch_size_samples=patch_size_samples,
                n_steps=args.n_steps,
                device=device
            )

            stem = f"{subject_id}_w{raw_idx}_{tier_tag}"

            gini, top_k_idx, top_k_scores = compute_channel_concentration(
                channel_attr, top_k=args.top_k_channels
            )
            is_focal = gini >= args.concentration_threshold
            verdict = "focal (clear winner)" if is_focal else "distributed (no clear winner)"
            print(
                f"  Channel concentration: Gini={gini:.3f} -> {verdict} "
                f"[threshold={args.concentration_threshold}]"
            )
            print(
                f"  Top {len(top_k_idx)} channels: "
                + ", ".join(f"ch{int(c)}={s:.4g}" for c, s in zip(top_k_idx, top_k_scores))
            )

            save_path = output_dir / f"{stem}_multi_granularity_attr.npz"
            np.savez_compressed(
                save_path,
                signal_attribution=signal_attr,  # [C, T]
                patch_attribution=patch_attr,    # [C, num_patches]
                channel_attribution=channel_attr,  # [C]
                raw_eeg=window_sample.squeeze(0),
                channel_gini=gini,
                top_k_channel_indices=top_k_idx,
                top_k_channel_scores=top_k_scores,
                is_focal=is_focal
            )
            print(f"Attribution dataset exported to {save_path}")

            plot_path = output_dir / f"{stem}_dashboard.png"
            plot_attribution_dashboard(
                raw_eeg=window_sample.squeeze(0),
                signal_attr=signal_attr,
                patch_attr=patch_attr,
                channel_attr=channel_attr,
                output_path=plot_path,
                target_channel_idx=args.target_channel_idx,
                sfreq=args.sfreq,
                concentration_threshold=args.concentration_threshold,
                top_k_channels=args.top_k_channels,
                gini=gini,
                top_k_idx=top_k_idx
            )


if __name__ == "__main__":
    main()
