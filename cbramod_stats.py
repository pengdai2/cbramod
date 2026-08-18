"""
cbramod_stats.py

A handful of pure-numpy statistics helpers shared across the cbramod script family, deliberately
kept in their OWN module rather than folded into cbramod_common.py: cbramod_common.py imports torch/
braindecode/safetensors/einops at module level, and several scripts in this family
(p09g_key_subject_feature_summary.py, p09j_band_covariation_check.py,
p22_ground_truth_band_power_comparison.py, and p23_capstone_figures.py, which reads p22's output) are
deliberately model-free -- they read already-saved CSVs and never touch the model, backbone, or a
GPU. That "no heavy ML dependency needed" property was validated directly (a throwaway venv without
torch installed) earlier in this project's investigation. Importing spearman_corr from
cbramod_common would silently reintroduce a torch/braindecode dependency into scripts that were
specifically designed not to need one -- this module exists so that doesn't happen.

cbramod_common.py itself imports spearman_corr from here (not the other way around), so every script
in the family -- heavy or light -- gets the exact same implementation either way.
"""

import numpy as np


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via plain rank + Pearson (no scipy.stats dependency needed) -- the
    single shared version of what used to be an identical copy duplicated across nine scripts
    (p09f/g/h/i/j/k, p14, p17, p22)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])
