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

try:
    import yasa
    HAS_YASA = True
except ImportError:
    HAS_YASA = False

from cbramod_common import add_log_filename_argument, build_e2e_classifier, seed_everything, setup_inference_cli_parser
from cbramod_utils import setup_logger
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
# Concentration metrics: is attribution focal or distributed, sustained or a
# transient spike?
# -----------------------------------------------------------------------------

def _gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a nonnegative array: 0.0 means attribution is spread
    perfectly evenly, 1.0 means it is concentrated entirely in a single
    element. Used as a cheap, threshold-able stand-in for "is there a clear
    winner, or is this spread out?" without requiring a human to eyeball a
    bar chart -- reused for both the channel (spatial) and patch (temporal)
    axes below.
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
      - gini: concentration coefficient (see `_gini_coefficient`). High ->
        one dominant channel (focal, spatially localized source). Low -> no
        single channel wins (distributed).
      - top_k_idx: indices of the top_k channels by attribution magnitude,
        descending
      - top_k_scores: their corresponding attribution scores
    """
    top_k = min(top_k, len(channel_attr))
    order = np.argsort(channel_attr)[::-1]
    top_k_idx = order[:top_k]
    return _gini_coefficient(channel_attr), top_k_idx, channel_attr[top_k_idx]


def compute_temporal_concentration(
    patch_attr: np.ndarray,
    top_k: int = 3
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Collapses `patch_attr` [Channels, Num_Patches] across channels (L2 norm)
    into a per-patch energy profile over time, then summarizes it exactly
    like `compute_channel_concentration` does for channels -- same metric,
    different axis:
      - gini: 0.0 -> attribution spread evenly across most/all patches: a
        *sustained* abnormality spanning the window. 1.0 -> concentrated in
        one or two patches: a *transient spike*. This directly operationalizes
        the "sustained (Tier 1) vs. spike (Tier 3)" distinction -- compare
        this value across a subject's windows from different tiers rather
        than eyeballing the patch heatmap each time.
      - top_k_idx: indices of the top_k patches (== seconds into the window,
        since each patch spans 1s) by attribution energy, descending.
      - top_k_scores: their corresponding patch energy.
    """
    patch_energy = np.linalg.norm(patch_attr, axis=0)  # [Num_Patches]
    top_k = min(top_k, len(patch_energy))
    order = np.argsort(patch_energy)[::-1]
    top_k_idx = order[:top_k]
    return _gini_coefficient(patch_energy), top_k_idx, patch_energy[top_k_idx]


# -----------------------------------------------------------------------------
# Morphology hotspot characterization: what does the model's top evidence
# actually look like?
# -----------------------------------------------------------------------------

# Coarse frequency bands used only to *tag* a hotspot for quick triage -- not
# a validated classifier. Always eyeball the plotted snippet before trusting
# a tag; dominant-frequency-via-FFT on a ~1s snippet is a rough estimate, and
# amplitude/duration heuristics will misfire on unusual morphology or noise.
MORPHOLOGY_TAG_RULES = (
    # (label, freq_lo, freq_hi, min_duration_sec, max_duration_sec, min_p2p_uv)
    ("sleep_spindle", 11.0, 16.0, 0.3, 3.0, 0.0),
    ("k_complex_or_slow_wave", 0.0, 2.0, 0.0, 1.5, 75.0),
    ("delta_slow_wave", 0.0, 4.0, 0.0, 1e9, 100.0),
    ("theta_burst", 4.0, 8.0, 0.0, 1e9, 0.0),
    ("alpha_burst", 8.0, 12.0, 0.0, 1e9, 0.0),
    ("sharp_transient_or_artifact", 20.0, 1e9, 0.0, 1e9, 100.0),
)


def _dominant_frequency(snippet: np.ndarray, sfreq: float) -> float:
    """FFT-based dominant frequency (Hz) of a 1D snippet, ignoring DC."""
    n = len(snippet)
    if n < 4:
        return 0.0
    windowed = snippet * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    if len(power) <= 1:
        return 0.0
    peak_idx = 1 + int(np.argmax(power[1:]))  # skip DC bin
    return float(freqs[peak_idx])


def _classify_morphology(dominant_freq_hz: float, duration_sec: float, peak_to_peak_uv: float) -> str:
    """Heuristic morphology tag from dominant frequency / duration / amplitude. See MORPHOLOGY_TAG_RULES."""
    for label, freq_lo, freq_hi, dur_lo, dur_hi, min_p2p in MORPHOLOGY_TAG_RULES:
        if freq_lo <= dominant_freq_hz <= freq_hi and dur_lo <= duration_sec <= dur_hi and peak_to_peak_uv >= min_p2p:
            return label
    return "unclassified"


def detect_yasa_events(eeg_channel: np.ndarray, sfreq: float) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Runs YASA's validated spindle (`spindles_detect`) and slow-wave
    (`sw_detect`) detectors once over a full channel's raw EEG, returning
    their summary DataFrames (or None if unavailable/nothing detected).

    Run once per window+channel and reused across all of that window's
    hotspots via `_match_yasa_event` -- these detectors rely on
    relative-power/envelope criteria that need more temporal context than a
    single ~1s hotspot snippet provides, so they must see the full window,
    not a truncated slice around one peak.
    """
    if not HAS_YASA:
        return None, None

    sp_df, sw_df = None, None
    try:
        sp = yasa.spindles_detect(eeg_channel, sf=sfreq, freq_sp=(11, 16), verbose=False)
        sp_df = sp.summary() if sp is not None else None
    except Exception as e:
        print(f"  [YASA] Spindle detection failed, falling back to heuristic for this channel: {e}")

    try:
        sw = yasa.sw_detect(eeg_channel, sf=sfreq, verbose=False)
        sw_df = sw.summary() if sw is not None else None
    except Exception as e:
        print(f"  [YASA] Slow-wave detection failed, falling back to heuristic for this channel: {e}")

    return sp_df, sw_df


def _match_yasa_event(
    peak_time_sec: float,
    sp_df: Optional[pd.DataFrame],
    sw_df: Optional[pd.DataFrame]
) -> Optional[Dict]:
    """
    Checks whether `peak_time_sec` falls inside any YASA-detected spindle or
    slow-wave/K-complex event's [Start, End] interval. Returns a dict with
    the validated tag, YASA's own duration/frequency/amplitude estimates
    (computed from its literature-benchmarked algorithm, not our FFT-peak
    heuristic), and the event's actual detected boundaries as
    window_start_sec/window_end_sec (so the plotted snippet shows the real
    detected event span rather than an arbitrary fixed half-span) -- or None
    if no detected event covers this timepoint.
    """
    if sp_df is not None and len(sp_df):
        hits = sp_df[(sp_df["Start"] <= peak_time_sec) & (peak_time_sec <= sp_df["End"])]
        if len(hits):
            row = hits.iloc[0]
            return {
                "morphology_tag": "sleep_spindle",
                "tag_source": "yasa_spindles_detect",
                "dominant_frequency_hz": float(row["Frequency"]),
                "duration_sec": float(row["Duration"]),
                "peak_to_peak_uv": float(row["Amplitude"]) if "Amplitude" in row else float("nan"),
                "window_start_sec": float(row["Start"]),
                "window_end_sec": float(row["End"]),
            }

    if sw_df is not None and len(sw_df):
        hits = sw_df[(sw_df["Start"] <= peak_time_sec) & (peak_time_sec <= sw_df["End"])]
        if len(hits):
            row = hits.iloc[0]
            return {
                "morphology_tag": "k_complex_or_slow_wave",
                "tag_source": "yasa_sw_detect",
                "dominant_frequency_hz": float(row["Frequency"]),
                "duration_sec": float(row["Duration"]),
                "peak_to_peak_uv": float(row["PTP"]) if "PTP" in row else float("nan"),
                "window_start_sec": float(row["Start"]),
                "window_end_sec": float(row["End"]),
            }

    return None


def characterize_morphology_hotspots(
    signal_attr: np.ndarray,
    raw_eeg: np.ndarray,
    sfreq: float,
    channel_idx: Optional[int] = None,
    top_k: int = 3,
    snippet_sec: float = 1.0,
    use_yasa: bool = True
) -> List[Dict]:
    """
    Finds the top_k highest (positive) attribution samples within
    `channel_idx` (defaults to the single most-attributed channel overall)
    and characterizes each one.

    Tagging priority: if `use_yasa` and YASA is installed, each hotspot is
    first checked against YASA's validated spindle/slow-wave detections for
    this channel (`detect_yasa_events` + `_match_yasa_event`) -- these use
    literature-benchmarked criteria over the full window, not a short
    snippet, and should be trusted over the fallback. Only when no detected
    event covers the hotspot (e.g. it's a sharp transient/artifact outside
    YASA's scope, or YASA isn't installed) do we fall back to the coarser
    dominant-frequency/duration/amplitude heuristic on a `snippet_sec`-wide
    window extracted around the peak. Each returned hotspot's "tag_source"
    field says which path produced it.

    This is still a triage aid, not ground truth -- pair with
    `plot_morphology_hotspots` and eyeball the returned snippet against the
    plotted trace rather than trusting any tag (YASA-sourced or heuristic)
    outright.

    Hotspots are found by repeatedly taking the highest remaining positive
    attribution sample, then "claiming" (excluding from further selection)
    its extraction window so consecutive hotspots don't just re-describe the
    same peak.
    """
    if channel_idx is None:
        channel_idx = int(np.argmax(np.linalg.norm(signal_attr, axis=1)))

    attr_1d = signal_attr[channel_idx]
    eeg_1d = raw_eeg[channel_idx]
    half_span = max(1, int(round(snippet_sec * sfreq / 2)))

    sp_df, sw_df = detect_yasa_events(eeg_1d, sfreq) if use_yasa else (None, None)

    pos_attr = np.maximum(attr_1d, 0)
    order = np.argsort(pos_attr)[::-1]
    claimed = np.zeros(len(attr_1d), dtype=bool)

    hotspots: List[Dict] = []
    for idx in order:
        if len(hotspots) >= top_k or pos_attr[idx] <= 0:
            break
        if claimed[idx]:
            continue

        start = max(0, idx - half_span)
        end = min(len(eeg_1d), idx + half_span)
        claimed[start:end] = True

        peak_time_sec = float(idx / sfreq)
        yasa_match = _match_yasa_event(peak_time_sec, sp_df, sw_df) if use_yasa else None

        if yasa_match is not None:
            hotspot = dict(yasa_match)
        else:
            snippet = eeg_1d[start:end]
            dominant_freq = _dominant_frequency(snippet, sfreq)
            duration_sec = (end - start) / sfreq
            peak_to_peak_uv = float(snippet.max() - snippet.min()) if len(snippet) else 0.0
            hotspot = {
                "morphology_tag": _classify_morphology(dominant_freq, duration_sec, peak_to_peak_uv),
                "tag_source": "heuristic",
                "dominant_frequency_hz": dominant_freq,
                "duration_sec": duration_sec,
                "peak_to_peak_uv": peak_to_peak_uv,
            }

        hotspot.update({
            "channel": channel_idx,
            "peak_sample_idx": int(idx),
            "peak_time_sec": peak_time_sec,
            "attribution": float(attr_1d[idx]),
        })
        # A YASA match already carries the real detected event's
        # window_start_sec/window_end_sec (see _match_yasa_event) -- only
        # fall back to the fixed half-span snippet bounds when there isn't one.
        hotspot.setdefault("window_start_sec", float(start / sfreq))
        hotspot.setdefault("window_end_sec", float(end / sfreq))
        hotspots.append(hotspot)

    return hotspots


def plot_morphology_hotspots(
    raw_eeg: np.ndarray,
    signal_attr: np.ndarray,
    hotspots: List[Dict],
    sfreq: float,
    output_path: Path
) -> None:
    """One row per hotspot: zoomed raw trace + positive attribution overlay, annotated with its morphology tag."""
    if not hotspots:
        print("  No positive-attribution hotspots found; skipping morphology plot.")
        return

    fig, axes = plt.subplots(len(hotspots), 1, figsize=(10, 2.6 * len(hotspots)), squeeze=False)

    for i, hotspot in enumerate(hotspots):
        ax = axes[i, 0]
        ch = hotspot["channel"]
        num_samples = raw_eeg.shape[1]
        start_sample = max(0, int(round(hotspot["window_start_sec"] * sfreq)))
        end_sample = min(num_samples, max(start_sample + 1, int(round(hotspot["window_end_sec"] * sfreq))))

        snippet_eeg = raw_eeg[ch, start_sample:end_sample]
        snippet_attr = np.maximum(signal_attr[ch, start_sample:end_sample], 0)
        snippet_time = np.arange(start_sample, end_sample) / sfreq

        ax.plot(snippet_time, snippet_eeg, color='black', linewidth=1.0)
        if snippet_attr.size and snippet_attr.max() > 0:
            attr_norm = snippet_attr / snippet_attr.max() * np.abs(snippet_eeg).max()
            ax.fill_between(snippet_time, 0, attr_norm, color='crimson', alpha=0.3)
        ax.axvline(hotspot["peak_time_sec"], color='crimson', linestyle='--', linewidth=1.0)

        source_note = "YASA-detected" if hotspot["tag_source"].startswith("yasa") else "heuristic"
        ax.set_title(
            f"Hotspot {i + 1}: Ch {ch} @ {hotspot['peak_time_sec']:.2f}s -- "
            f"{hotspot['morphology_tag']} [{source_note}] "
            f"(dom.freq={hotspot['dominant_frequency_hz']:.1f}Hz, "
            f"dur={hotspot['duration_sec']:.2f}s, p2p={hotspot['peak_to_peak_uv']:.1f}uV)",
            fontsize=10
        )
        ax.set_ylabel('Amplitude (µV)', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

    axes[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Morphology hotspot plot saved to: {output_path}")


def scan_yasa_events_across_channels(
    raw_eeg: np.ndarray,
    sfreq: float,
    channels: Optional[List[int]] = None
) -> Dict[int, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Runs `detect_yasa_events` for every channel in `channels` (default: all
    channels in `raw_eeg`), not just whichever channels Panel 1 happens to
    display. Panel 1 only ever shows the top-attributed channel(s); a
    genuine spindle/slow-wave on some other channel would otherwise never
    even be checked for, let alone shown. Returns {channel_idx: (spindle_df,
    slow_wave_df)}.
    """
    if channels is None:
        channels = list(range(raw_eeg.shape[0]))
    return {ch: detect_yasa_events(raw_eeg[ch], sfreq) for ch in channels}


def summarize_yasa_channel_scan(
    events_by_channel: Dict[int, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]
) -> List[Dict]:
    """Per-channel spindle/slow-wave counts from a full scan, one row per channel, sorted by channel index."""
    summary = []
    for ch in sorted(events_by_channel.keys()):
        sp_df, sw_df = events_by_channel[ch]
        summary.append({
            "channel": ch,
            "num_spindles": len(sp_df) if sp_df is not None else 0,
            "num_slow_waves": len(sw_df) if sw_df is not None else 0
        })
    return summary


def _annotate_yasa_events(ax: plt.Axes, sp_df: Optional[pd.DataFrame], sw_df: Optional[pd.DataFrame]) -> bool:
    """
    Shades every YASA-detected spindle/slow-wave span on `ax`, regardless of
    whether it overlaps a high-attribution hotspot -- a detected event the
    model *didn't* attend to is still informative context (e.g. "there was
    a spindle right here and the model's attribution ignored it entirely").
    Returns whether anything was actually drawn, so the caller only adds a
    legend when there's something to label.
    """
    drew_anything = False

    if sp_df is not None and len(sp_df):
        for i, (_, row) in enumerate(sp_df.iterrows()):
            ax.axvspan(
                row["Start"], row["End"], color='gold', alpha=0.25,
                label='YASA Spindle' if i == 0 else None
            )
        drew_anything = True

    if sw_df is not None and len(sw_df):
        for i, (_, row) in enumerate(sw_df.iterrows()):
            ax.axvspan(
                row["Start"], row["End"], color='steelblue', alpha=0.2,
                label='YASA Slow Wave' if i == 0 else None
            )
        drew_anything = True

    return drew_anything


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
    top_k_idx: Optional[np.ndarray] = None,
    temporal_concentration_threshold: float = 0.5,
    top_k_patches: int = 3,
    temporal_gini: Optional[float] = None,
    top_k_patch_idx: Optional[np.ndarray] = None,
    show_yasa_events: bool = True,
    yasa_scan_all_channels: bool = True,
    yasa_scan_json_path: Optional[Path] = None
):
    """
    Generates a visual diagnostic dashboard comparing raw EEG signals,
    signal-level saliency, and CBraMod patch grid importance.

    Panel 1 shows one small-multiple row per channel being displayed --
    a single channel when attribution is focal (or --target-channel-idx is
    given), or the top `top_k_channels` channels stacked in separate rows
    (not overlaid on one axis) when it's distributed, so multi-channel cases
    stay readable instead of piling multiple colored fills on top of each
    other. When `show_yasa_events` and YASA is installed, every displayed
    row also shades that channel's YASA-detected spindle/slow-wave spans
    across the *entire* window -- independent of
    `characterize_morphology_hotspots`'s hotspot matching, so a detected
    event shows up here even on windows/tiers where morphology
    characterization wasn't run, or where it didn't line up with a
    high-attribution sample.

    Panel 1 only ever shows the top-attributed channel(s), though -- a real
    spindle/slow-wave on some *other* channel wouldn't be checked for at all
    if we only scanned displayed channels. So when `yasa_scan_all_channels`
    (default True), every channel in `raw_eeg` is scanned, not just the
    displayed ones: channels not shown as a full Panel 1 row still get a
    marker on Panel 3's bar chart at their own channel index if they have a
    detected event, and (if `yasa_scan_json_path` is given) the full
    per-channel spindle/slow-wave counts are saved there. Set to False to
    only scan displayed channels (previous behavior, cheaper -- scanning all
    channels runs two detectors per channel, so cost scales with channel
    count regardless of how many are actually plotted).

    Panel 2's patch heatmap is annotated with the temporal-concentration
    verdict (sustained vs. spike-localized) and marks the top attributed
    patches with vertical guides.

    `gini`/`top_k_idx` (channel) and `temporal_gini`/`top_k_patch_idx`
    (patch/time) can be passed in pre-computed (e.g. from `main()`, which
    already needs them for logging/export) to avoid recomputing; if omitted
    they're derived here.
    """
    time_axis = np.arange(raw_eeg.shape[1]) / sfreq
    num_channels, num_patches = patch_attr.shape

    if gini is None or top_k_idx is None:
        gini, top_k_idx, _ = compute_channel_concentration(channel_attr, top_k=top_k_channels)
    is_focal = gini >= concentration_threshold

    if temporal_gini is None or top_k_patch_idx is None:
        temporal_gini, top_k_patch_idx, _ = compute_temporal_concentration(patch_attr, top_k=top_k_patches)
    is_temporally_focal = temporal_gini >= temporal_concentration_threshold

    # --- Panel 1 channel selection ---
    if target_channel_idx is not None:
        # Explicit user override always wins, regardless of concentration.
        plot_channels = [target_channel_idx]
    elif is_focal:
        plot_channels = [int(top_k_idx[0])]
    else:
        plot_channels = [int(c) for c in top_k_idx]

    # Scan for YASA events once, up front -- across all channels by default
    # (see docstring), not just `plot_channels`, so events elsewhere aren't
    # silently never even checked for.
    events_by_channel: Dict[int, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]] = {}
    if show_yasa_events and HAS_YASA:
        scan_channels = list(range(num_channels)) if yasa_scan_all_channels else plot_channels
        events_by_channel = scan_yasa_events_across_channels(raw_eeg, sfreq, channels=scan_channels)

    num_channel_rows = len(plot_channels)
    total_rows = num_channel_rows + 2  # + patch heatmap + channel bar chart
    height_ratios = [1.0] * num_channel_rows + [1.4, 1.2]

    fig = plt.figure(figsize=(14, 2.2 * num_channel_rows + 6.5))
    gs = fig.add_gridspec(total_rows, 1, height_ratios=height_ratios, hspace=0.6)

    channel_axes = [fig.add_subplot(gs[i, 0]) for i in range(num_channel_rows)]
    ax_patch = fig.add_subplot(gs[num_channel_rows, 0])
    ax_bar = fig.add_subplot(gs[num_channel_rows + 1, 0])

    # --- Panel 1: Raw EEG Trace with Overlay Signal Attribution (one row per channel) ---
    for i, ch in enumerate(plot_channels):
        ax = channel_axes[i]
        eeg_signal = raw_eeg[ch]
        attr_signal = signal_attr[ch]

        ax.plot(time_axis, eeg_signal, color='black', alpha=0.75, linewidth=0.9, label='Raw EEG', zorder=3)
        pos_attr = np.maximum(0, attr_signal)
        if pos_attr.max() > 0:
            pos_attr_norm = pos_attr / pos_attr.max() * np.abs(eeg_signal).max()
            ax.fill_between(time_axis, 0, pos_attr_norm, color='crimson', alpha=0.3, label='Attribution', zorder=2)

        drew_events = False
        if ch in events_by_channel:
            sp_df, sw_df = events_by_channel[ch]
            drew_events = _annotate_yasa_events(ax, sp_df, sw_df)

        rank_note = " (Highest)" if i == 0 and num_channel_rows > 1 else ""
        ax.set_ylabel(f'Ch {ch}{rank_note}\n(µV)', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        if drew_events:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
        if i < num_channel_rows - 1:
            ax.set_xticklabels([])

    if target_channel_idx is not None:
        header = f'Signal-Level Attribution (Channel {target_channel_idx}, User-Selected)'
    elif is_focal:
        header = f'Signal-Level Attribution (Channel {plot_channels[0]}, Highest Attribution, Gini={gini:.2f})'
    else:
        header = f'Signal-Level Attribution (Top {num_channel_rows} Channels, No Clear Winner, Gini={gini:.2f})'
    channel_axes[0].set_title(header, fontsize=12)
    channel_axes[-1].set_xlabel('Time (s)')

    # --- Panel 2: CBraMod Spatial-Temporal Patch Heatmap ---
    im = ax_patch.imshow(
        patch_attr,
        aspect='auto',
        cmap='magma',
        origin='lower',
        extent=[0, time_axis[-1], 0, num_channels]
    )
    for patch_idx in top_k_patch_idx:
        ax_patch.axvline(patch_idx + 0.5, color='cyan', linestyle='--', linewidth=1.0, alpha=0.8)
    temporal_verdict = "Spike-Localized" if is_temporally_focal else "Sustained"
    ax_patch.set_ylabel('EEG Channel Index')
    ax_patch.set_title(
        f'CBraMod Token Patch Importance Heatmap (Channels x Time Patches) -- '
        f'{temporal_verdict} (Temporal Gini={temporal_gini:.2f})'
    )
    fig.colorbar(im, ax=ax_patch, orientation='vertical', label='Patch Importance (L2 Norm)')

    # --- Panel 3: Global Channel Importance ---
    ax_bar.bar(range(num_channels), channel_attr, color='navy', alpha=0.7)
    ax_bar.set_xlabel('EEG Channel Index')
    ax_bar.set_ylabel('Attribution L2 Norm')
    ax_bar.set_title('Global Channel Importance Profile')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    # Mark every channel with a detected YASA event here too, regardless of
    # attribution rank -- this is the one place a spindle/slow-wave on a
    # channel NOT shown as a full Panel 1 row still becomes visible, rather
    # than only ever being checked for on the (up to top_k_channels)
    # displayed channels.
    if events_by_channel:
        marker_headroom = max(channel_attr.max(), 1e-9) * 0.08
        drew_bar_spindle, drew_bar_sw = False, False
        for ch, (sp_df, sw_df) in events_by_channel.items():
            n_sp = len(sp_df) if sp_df is not None else 0
            n_sw = len(sw_df) if sw_df is not None else 0
            marker_y = channel_attr[ch] + marker_headroom
            if n_sp:
                ax_bar.scatter(
                    ch, marker_y, marker='*', s=70, color='gold', edgecolors='black', linewidths=0.5,
                    zorder=5, label='YASA Spindle Detected' if not drew_bar_spindle else None
                )
                drew_bar_spindle = True
                marker_y += marker_headroom
            if n_sw:
                ax_bar.scatter(
                    ch, marker_y, marker='v', s=55, color='steelblue', edgecolors='black', linewidths=0.5,
                    zorder=5, label='YASA Slow Wave Detected' if not drew_bar_sw else None
                )
                drew_bar_sw = True
        if drew_bar_spindle or drew_bar_sw:
            ax_bar.legend(loc='upper right', fontsize=8)

        undisplayed_with_events = sorted(
            ch for ch, (sp_df, sw_df) in events_by_channel.items()
            if ch not in plot_channels and ((sp_df is not None and len(sp_df)) or (sw_df is not None and len(sw_df)))
        )
        if undisplayed_with_events:
            print(
                f"  [YASA] Detected event(s) on channel(s) not shown in Panel 1: {undisplayed_with_events} "
                f"-- see Panel 3 markers" + (f" / {yasa_scan_json_path}" if yasa_scan_json_path else "")
            )

        if yasa_scan_json_path is not None:
            with open(yasa_scan_json_path, "w") as f:
                json.dump(summarize_yasa_channel_scan(events_by_channel), f, indent=2)
            print(f"  Full per-channel YASA scan saved to: {yasa_scan_json_path}")

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
             "so --top-k-channels channels are shown in separate rows in Panel 1 instead."
    )
    attr_group.add_argument(
        "--top-k-channels", type=int, default=3,
        help="Number of channels to show (one per row) in Panel 1 when attribution is distributed rather than focal."
    )
    attr_group.add_argument(
        "--temporal-concentration-threshold", type=float, default=0.5,
        help="Gini coefficient (0-1) on patch-level (time-collapsed) attribution above which a window is "
             "treated as 'spike-localized' (attribution concentrated in one or two patches). Below this, "
             "it's treated as 'sustained' (spread across most/all patches)."
    )
    attr_group.add_argument(
        "--top-k-patches", type=int, default=3,
        help="Number of highest-attribution patches (time positions) to mark on the Panel 2 heatmap."
    )
    attr_group.add_argument(
        "--top-k-hotspots", type=int, default=3,
        help="Number of morphology hotspots (highest-attribution raw-signal peaks) to characterize and plot "
             "per window, for windows whose tier matches --morphology-tiers. Set to 0 to disable."
    )
    attr_group.add_argument(
        "--morphology-tiers", type=str, default="top",
        help="Comma-separated tier filter (substring-matched, like --tiers) selecting which tiers get "
             "morphology hotspot characterization. Default: 'top' (matches 'Tier 1: Top Drivers' only) -- "
             "the model's most confident evidence is the natural place to start looking for the underlying "
             "morphology (e.g. sleep spindles, sharp waves) driving the prediction."
    )
    attr_group.add_argument(
        "--hotspot-snippet-sec", type=float, default=1.0,
        help="Width (seconds) of the raw-EEG snippet extracted around each morphology hotspot peak, used "
             "only as a fallback when YASA doesn't detect a validated event covering that peak (or "
             "--no-yasa-detectors is set)."
    )
    attr_group.add_argument(
        "--no-yasa-detectors", dest="use_yasa_detectors", action="store_false", default=True,
        help="Disable all use of YASA's validated spindle/slow-wave detectors (yasa.spindles_detect/"
             "sw_detect): stops shading detected event spans on every dashboard's Panel 1 (shown "
             "regardless of tier, whether or not they overlap a hotspot), and stops checking morphology "
             "hotspots against them (falling back to the dominant-frequency/duration/amplitude heuristic "
             "for every hotspot instead). Has no effect if YASA isn't installed."
    )
    attr_group.add_argument(
        "--no-yasa-scan-all-channels", dest="yasa_scan_all_channels", action="store_false", default=True,
        help="Only run YASA detectors on the channel(s) Panel 1 actually displays, instead of every "
             "channel. By default all channels are scanned so an event on a channel that wasn't among the "
             "top-attributed ones still shows up as a marker on Panel 3 (and in the saved per-channel scan) "
             "rather than never being checked for at all -- pass this flag to skip that (cheaper: scanning "
             "runs two detectors per channel, so cost scales with total channel count, not just how many "
             "are displayed)."
    )
    add_log_filename_argument(parser, __file__)

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
    logger = setup_logger(args.log_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./attribution_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CBraMod tokenizes each 1-second slice of signal as one patch, so a
    # window's duration in seconds is simply `num_patches`, and each patch
    # spans exactly `sfreq` samples — no separate window-length flag needed.
    patch_size_samples = int(round(args.sfreq))
    window_sec = float(args.num_patches)
    morphology_tier_filter = _resolve_tier_filter(args.morphology_tiers)
    # Two independent uses of YASA: showing every detected spindle/slow-wave
    # span on the dashboard (every window, regardless of tier or whether it
    # lines up with a hotspot) vs. using detections to tag morphology
    # hotspots (only for --morphology-tiers windows, only if --top-k-hotspots
    # > 0). Decoupled so disabling hotspot characterization doesn't also
    # hide the dashboard annotations, and vice versa.
    show_yasa_events = args.use_yasa_detectors
    use_yasa_for_hotspots = args.use_yasa_detectors and args.top_k_hotspots > 0
    if args.use_yasa_detectors and not HAS_YASA:
        print(
            "  [Warning] --no-yasa-detectors was not set but YASA is not installed; dashboards won't show "
            "YASA event spans, and morphology hotspots will use the dominant-frequency/duration/amplitude "
            "heuristic instead of YASA's validated spindle/slow-wave detectors."
        )

    # Metadata-first architecture resolution + deterministic (checkpoint_kind-driven) state dict
    # loading -- same helper as p09/p09c/p09e.
    model, _ckpt = build_e2e_classifier(args, device, logger)

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
            raw_eeg = window_sample.squeeze(0)

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

            temporal_gini, top_k_patch_idx, top_k_patch_scores = compute_temporal_concentration(
                patch_attr, top_k=args.top_k_patches
            )
            is_temporally_focal = temporal_gini >= args.temporal_concentration_threshold
            temporal_verdict = "spike-localized" if is_temporally_focal else "sustained"
            print(
                f"  Temporal concentration: Gini={temporal_gini:.3f} -> {temporal_verdict} "
                f"[threshold={args.temporal_concentration_threshold}]"
            )
            print(
                f"  Top {len(top_k_patch_idx)} patches (seconds into window): "
                + ", ".join(f"t={int(p)}s={s:.4g}" for p, s in zip(top_k_patch_idx, top_k_patch_scores))
            )

            save_path = output_dir / f"{stem}_multi_granularity_attr.npz"
            np.savez_compressed(
                save_path,
                signal_attribution=signal_attr,  # [C, T]
                patch_attribution=patch_attr,    # [C, num_patches]
                channel_attribution=channel_attr,  # [C]
                raw_eeg=raw_eeg,
                channel_gini=gini,
                top_k_channel_indices=top_k_idx,
                top_k_channel_scores=top_k_scores,
                is_focal=is_focal,
                temporal_gini=temporal_gini,
                top_k_patch_indices=top_k_patch_idx,
                top_k_patch_scores=top_k_patch_scores,
                is_temporally_focal=is_temporally_focal
            )
            print(f"Attribution dataset exported to {save_path}")

            plot_path = output_dir / f"{stem}_dashboard.png"
            plot_attribution_dashboard(
                raw_eeg=raw_eeg,
                signal_attr=signal_attr,
                patch_attr=patch_attr,
                channel_attr=channel_attr,
                output_path=plot_path,
                target_channel_idx=args.target_channel_idx,
                sfreq=args.sfreq,
                concentration_threshold=args.concentration_threshold,
                top_k_channels=args.top_k_channels,
                gini=gini,
                top_k_idx=top_k_idx,
                temporal_concentration_threshold=args.temporal_concentration_threshold,
                top_k_patches=args.top_k_patches,
                temporal_gini=temporal_gini,
                top_k_patch_idx=top_k_patch_idx,
                show_yasa_events=show_yasa_events,
                yasa_scan_all_channels=args.yasa_scan_all_channels,
                yasa_scan_json_path=output_dir / f"{stem}_yasa_channel_scan.json" if show_yasa_events else None
            )

            # Morphology hotspot characterization -- only for tiers matching
            # --morphology-tiers (default: Tier 1 / Top Drivers), since that's
            # the model's most confident evidence and the natural starting
            # point for "what does this actually look like."
            run_morphology = args.top_k_hotspots > 0 and (
                morphology_tier_filter is None or task["tier"] in morphology_tier_filter
            )
            if run_morphology:
                hotspots = characterize_morphology_hotspots(
                    signal_attr=signal_attr,
                    raw_eeg=raw_eeg,
                    sfreq=args.sfreq,
                    channel_idx=int(top_k_idx[0]),
                    top_k=args.top_k_hotspots,
                    snippet_sec=args.hotspot_snippet_sec,
                    use_yasa=use_yasa_for_hotspots
                )
                print(f"  Morphology hotspots (Ch {int(top_k_idx[0])}):")
                for h in hotspots:
                    print(
                        f"    t={h['peak_time_sec']:.2f}s -> {h['morphology_tag']} [{h['tag_source']}] "
                        f"(dom.freq={h['dominant_frequency_hz']:.1f}Hz, dur={h['duration_sec']:.2f}s, "
                        f"p2p={h['peak_to_peak_uv']:.1f}uV, attribution={h['attribution']:.4g})"
                    )

                hotspots_json_path = output_dir / f"{stem}_morphology_hotspots.json"
                with open(hotspots_json_path, "w") as f:
                    json.dump(hotspots, f, indent=2)
                print(f"  Morphology hotspots exported to {hotspots_json_path}")

                hotspots_plot_path = output_dir / f"{stem}_morphology_hotspots.png"
                plot_morphology_hotspots(
                    raw_eeg=raw_eeg,
                    signal_attr=signal_attr,
                    hotspots=hotspots,
                    sfreq=args.sfreq,
                    output_path=hotspots_plot_path
                )


if __name__ == "__main__":
    main()
