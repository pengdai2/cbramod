# Does the Model's Clinical Prediction Rest on Real Signal? A Causal Investigation of Sigma-Band Power

## 1. Motivation

CBraMod-based probing on our EEG cohort produces a subject-level clinical prediction (patient vs.
control) by pooling window-level probabilities (85th percentile of the window scores across a
subject's recording). Correlational analyses had repeatedly flagged the sigma band (~12-16 Hz,
spindle range) as behaving differently from the other canonical bands: it was the one band whose
relative power didn't fall into the mutually-reinforcing delta/theta/alpha cluster, and it showed
a distinct relationship with predicted probability once the underlying signal pipeline was fixed
(see Background). But correlation alone can't tell us whether the model is actually *using* sigma
power as evidence, or whether sigma power is merely a marker of some other confound (subject
identity, recording quality, an artifact of relative/compositional band power, etc.) that happens
to move together with the prediction. The goal of this investigation was to walk back from the
subject-level decision to a causal claim: does deliberately perturbing sigma power, and nothing
else, change what the model predicts — and does that effect survive at the level pooling actually
operates at (a whole subject's recording), not just at the level of an individual window?

## 2. Technical Background

**Pipeline.** Each window's raw EEG passes through the CBraMod backbone, is pooled across channels
(channel-mean), and z-scored per-window-per-channel before featurization. A window-level
probability is produced per window, and a subject-level decision is produced by taking a fixed
percentile (p85) of all the subject's window scores.

**Two confounds fixed before this investigation could be trusted.** First, a referencing bug meant
100% of subjects were silently falling back to Common Average Reference (CAR) instead of the
intended linked-earlobe (A1/A2) reference, because A1/A2 channels were being dropped from the pick
list before the referencing step ever looked for them. CAR referencing over near-uniform cortical
signal tends to cancel out real amplitude differences, producing artificially uniform, low-variance
channel-mean signals — this was silently degrading every absolute-power and YASA-based
(spindle/slow-wave detection) analysis run before the fix. Second, *relative* (ratio) band power is
compositionally constrained — five bands summing to ~1 mechanically forces anti-correlation with
whichever band dominates, independent of any real physiology. Absolute power, computed on the
properly-referenced signal, avoids this closure artifact and is what all analyses below use.

**Why percentile pooling complicates causal reasoning.** Pooling via percentile is a nonlinear,
whole-distribution-dependent statistic, not a linear average. Leave-one-out analysis showed that,
holding a subject's window scores fixed, only windows immediately adjacent to the 85th-percentile
rank have any marginal influence on the pooled score — most windows have zero leave-one-out
contribution. This means "which windows matter" is not a fixed, perturbation-invariant property
that can be read off the baseline distribution; it was tempting, but wrong, to reason from
"baseline-influential windows" about what a broader perturbation would do.

## 3. Investigation

**Corroborating evidence from an independent detector (YASA).** Before moving to intervention,
relative band power's sigma-specific signal was cross-checked against a completely different
measurement approach: YASA's validated spindle (`spindles_detect`) and slow-wave (`sw_detect`)
event detectors run on the properly-referenced signal, correlating event *counts* per window
against predicted probability. If sigma power's relationship with probability were a spectral
artifact rather than something tied to real spindle activity, an independent, waveform-based event
detector would have no reason to agree with it. It did: spindle count correlated negatively with
probability consistently across both validation and test sets, both pooled and within-subject
(validation: pooled r = -0.123 n=6800, within-subject mean r = -0.060/median r = -0.037, n=34
subjects; test: pooled r = -0.157 n=7354, within-subject mean r = -0.061/median r = -0.031, n=36
subjects) — the same sign, at a comparable magnitude, as sigma relative power's own within-subject
correlation (median r ≈ -0.10 to -0.16 across the two sets). Slow-wave count showed no such
agreement: correlations were small and inconsistent in sign across the two sets (pooled r = +0.058
on validation vs. -0.037 on test; within-subject mean r ≈ +0.01 on both), i.e. essentially null. This
asymmetry — spindles (sigma-band events) converging with the sigma story, slow waves (delta-band
events) not converging with anything — is itself informative: it argues the sigma finding tracks
something real and specific to spindle-band activity, rather than being a generic artifact that
would be expected to show up equally in any morphological event count.

**Window-level causal test.** For sampled windows, sigma-band power was scaled by a controlled
factor (0.5x-1.5x) via bandpass filtering, with total signal energy preserved to keep the perturbed
window inside the model's training distribution (avoiding a "went out of distribution" confound).
This established that scaling sigma power up moves individual window-level probabilities down in a
consistent, dose-proportional way — a direct causal effect, not just a correlation.

**Subject-level causal test (does pooling preserve or destroy this effect?).** The window-level
result doesn't guarantee anything at the subject level, since p85 pooling could in principle be
insensitive to a change that doesn't land near the percentile rank. Perturbing *all* windows of a
subject and re-pooling showed 95% of subjects moved in the same (negative) direction, with a strong
linear fit (R² ≈ 0.97) between pooled score and scale factor, and a "propagation ratio" (pooled-score
shift ÷ naive-mean-of-window-scores shift) with a median near 0.93 — meaning percentile pooling
transmits about as much of the window-level effect as plain averaging would, i.e. pooling is not
silently discarding the signal.

**The dose-response sharpening.** Perturbing *all* windows doesn't distinguish a genuine
propagating effect from a mechanical artifact of moving essentially the entire recording at once.
To sharpen this, perturbation was applied to a graded, nested sequence of random subsets (25%, 50%,
75%, 100% of a subject's windows, each a strict superset of the smaller fraction, so the swept
variable is cleanly "how much of the recording changed"). An earlier attempt to explain results via
"did perturbation hit the baseline-influential windows" was abandoned once it became clear that
percentile pooling's dependence on the whole sorted distribution means "influential windows" isn't
a fixed, perturbation-invariant concept — a window with zero baseline influence can still end up
determining the percentile once the distribution shifts.

## 4. Key Results

- **Independent corroboration, and a telling asymmetry**: YASA-detected spindle count (a
  waveform-based detector, not a spectral-power measurement) correlates negatively with predicted
  probability on both validation and test sets, pooled and within-subject, converging with the
  sigma relative-power finding — while slow-wave count shows no such consistency (sign flips
  between validation and test, near-zero within-subject). Two independent measurement methods
  agreeing specifically on sigma/spindles, and specifically failing to agree on delta/slow-waves,
  argues the sigma finding is tracking something real rather than a generic artifact of either
  measurement approach.
- **Direction and reliability**: increasing sigma power reliably decreases the subject-level
  predicted probability in 35/37 subjects (94.6%), and this fraction is *bit-for-bit identical* at
  every perturbation dose tested (25/50/75/100%) — the same two subjects buck the trend
  consistently, arguing this is a real per-subject exception, not sampling noise.
- **Dose-response confirms a real, non-artifactual effect**: raw effect size (subject-level slope)
  scales down almost exactly proportionally with the perturbed fraction (-0.0551 at 100% →
  -0.0135 at 25%, a ~4x change tracking a 4x change in fraction), consistent with a model where
  sigma power's effect is roughly uniform across a subject's windows and perturbing a random subset
  translates the whole score distribution by a proportionally diluted amount.
- **Pooling efficiency is dose-invariant**: the propagation ratio's median stays essentially flat
  (0.91-0.94) across the entire 25%-100% range (Spearman ≈ +0.08, i.e. no meaningful trend) — the
  *fraction* of window-level signal that reaches the subject-level decision doesn't degrade as less
  of the recording is perturbed. (The mean of this ratio is noisier — a few subjects with a small
  but nonzero denominator produce outsized ratios — the median is the reliable summary.)
- **Resolving an apparent contradiction**: this dose-invariant propagation is compatible with the
  earlier finding that most windows have zero baseline leave-one-out influence, because under a
  continuously shifting distribution, a *different* window keeps occupying the percentile rank at
  each step, each reflecting the same aggregate shift — no single window has to carry the whole
  effect for the percentile to track a distributional shift about as faithfully as the mean does.

**Bottom line**: this is genuine causal evidence, not merely correlational, that the model has
learned to treat elevated sigma-band power as evidence against the patient class, that this
relationship is not an artifact of the specific "perturb everything at once" setup, and that
percentile pooling does not silently discard this signal on its way from window-level to
subject-level decisions.

## 5. Next Steps

With the causal chain from band-level perturbation to subject-level decision now established and
validated, the natural next question is whether a *learned* aggregation over windows would do
better than the fixed p85 heuristic — motivating a move to attention-based multiple-instance
learning (MIL), in two candidate forms:

- **Option A**: attention over frozen window-level probabilities (backbone stays frozen; only the
  aggregation step becomes learned instead of a fixed percentile).
- **Option B**: attention over frozen window-level embeddings, with the aggregation itself learning
  which windows to weight, rather than assuming a fixed percentile rank is always the right summary
  statistic.

Both keep the backbone frozen to isolate the effect of aggregation choice; the open design question
is how much capacity/data is warranted per option before this stops being a lightweight follow-up to
the frozen-backbone probing line of work.

---

## Appendix A: Data Preparation & Cleansing

**Referencing.** Recordings are re-referenced (A1/A2 linked-earlobe reference, matching the
intended clinical convention) before any downstream slicing or feature extraction. A bug meant
A1/A2 channels were dropped from the pick list before the referencing step could find them, so
every subject silently fell back to Common Average Reference (CAR) regardless of whether A1/A2
electrodes existed in the recording. Fixed by discovering A1/A2 in the original channel list
(using the same name-cleaning logic as the referencing step itself) before picking down to the
recording's standard 64-channel montage, including them in the filter/reference pick list, and only
dropping them after referencing completes. The referencing function's CAR fallback branch also had
a `try/except` that silently swallowed failures and returned unreferenced data; this now propagates
any failure loudly instead.

**Bad-channel handling.** Channels are checked against their expected 3D scalp coordinates and
spatial-neighborhood consistency; channels that fail are interpolated where possible. A spatial
adjacency check's exception handler previously defaulted to "assume safe" (`return True`) on
failure; changed to fail closed (`return False`) with a logged warning. The function that used to
return one undifferentiated "bad channels" list now distinguishes interpolated channels from
channels skipped outright (missing coordinates, or the subject rejected entirely) — these had been
mislabeled identically before. A per-window `active_channel_mask` (boolean, aligned to the standard
64-channel montage) is now persisted in each subject's metadata, recording which channels actually
contributed real (non-zero-padded) data.

**Window-level artifact gatekeeping.** Windows are rejected (`EXTREME_ARTIFACT`) if their amplitude
exceeds a threshold (±500 µV clip triggers a `num_samples_clipped` diagnostic; a stricter std-based
ceiling flags the window outright). A hypothesis that this ceiling disproportionately penalizes
real high-amplitude N3 slow-wave content (rather than genuine artifact) was tested directly by
comparing `EXTREME_ARTIFACT` rejection rate by sleep stage across the cohort
(`p02d_rejection_reason_by_stage.py`): WAKE has the highest rate (15.97%) by a wide margin, N3 the
lowest (5.15%) — refuting the hypothesis. The gatekeeping is not conflating slow-wave morphology
with artifact.

**Subject-level gatekeeping.** `evaluate_subject_quality()` screens subjects post-slicing (e.g. on
overall rejection rate per stage). An N3-specific check was found to have a loophole — it trivially
passes for subjects with zero N3 windows — noted but not yet closed, since the artifact-rejection
concern it was meant to guard against was independently refuted above.

**Slicing strategies.** Two window-extraction strategies exist (Strategy A "macro", Strategy B
"micro"); both were unified to use the same metadata key names
(`bad_channels_interpolated`/`bad_channels_not_interpolated`) across their valid/rejected output
branches. Strategy B had a latent bug — a dead `if HAS_YASA:` guard referencing an undefined name
left over from an earlier refactor, and a missing `window_idx` field that silently broke a
downstream helper (`extract_valid_window_indices()`) expecting it — both fixed.

**Verification after re-slicing.** Because the referencing fix required re-slicing the entire
cohort, `p02x_verify_reslice_diff.py` was built to batch-compare old vs. new sliced output
(max/mean absolute difference, window counts, per-window bad-channel lists) — used first to confirm
a small, incidental "changed" subset (54/304 subjects) was pure float32 rounding noise (~1 ULP)
unrelated to any deliberate change, and later to confirm the actual referencing fix produced the
expected, much larger, systematic differences.

## Appendix B: Pipeline Configuration & Rationale

**Backbone and featurization.** CBraMod produces per-channel, per-window embeddings; these are
pooled across channels via a simple channel-mean, then z-scored per-window-per-channel before being
fed to the classification head. Z-scoring destroys absolute amplitude information at the
featurization step (a design point that turned out to matter — see the compositional band-power
confound in the main writeup), but the original scale is reconstructible from persisted
`norm_mean_uv`/`norm_std_uv` values when absolute-power analysis is needed downstream.

**Subject-level aggregation.** Window-level probabilities are pooled into one subject-level score
via a fixed 85th-percentile statistic (`p85_score`) rather than a mean, on the rationale that a
clinically meaningful pattern may only manifest in a minority of a recording's windows and a mean
would dilute it. This choice is precisely what motivated the causal-pooling-propagation analysis in
the main writeup, and what the proposed attention-based MIL work aims to replace with a learned
alternative.

**Checkpoint selection during training.** Validation-set checkpoint selection between epochs
required both F1 and AUC to be considered, on a validation cohort small enough (~35-40 subjects)
that either metric alone is vulnerable to noise-chasing. The selection rule went through several
iterations: an initial F1-primary/AUC-tiebreak design was rejected as asymmetric and unprincipled
(an F1 uptick bought at a real AUC cost isn't obviously an improvement); replaced with strict Pareto
dominance (neither metric may regress, at least one must strictly improve); then refined with an
explicit exception for a large gain in one metric alongside a small dip in the other
(`min_large_gain=0.02`, `max_small_dip=0.01`, both with float-precision tolerance), since strict
Pareto alone was seen (in a real training run) to reject a checkpoint with a substantial AUC gain
over a negligible F1 dip.

**Feature-cache architecture.** Backbone feature extraction was originally performed separately for
each of train/val/test/each CV fold, which was both slow (redundant extraction) and awkward
(temporary per-fold cache files). This was consolidated into a single, one-time extraction pass
over the entire cohort's master manifest (`p08a_extract_features.py`), with a
`CachedFeatureSubjectDataset` (plus a `flatten_cached_feature_dataset()` helper to recover
flat window-level arrays from a subject-grouped view) providing subject-filtered views of that one
cache for any subset of subjects needed downstream. Cross-validation (`StratifiedGroupKFold`) pools
subjects from this master cache, but must explicitly exclude test-set subjects from that pool
before splitting — the master cache spans the whole cohort including test, so building the CV pool
without this exclusion would leak test subjects into training folds. This was caught and fixed by
restricting the pool to the union of the training/validation manifests' subjects before computing
the fold split, with the excluded count logged for visibility.

**Band-power perturbation mechanics.** Perturbation scales a target band's power via zero-phase
(`sosfiltfilt`) bandpass filtering at a controlled scale factor, with an option to preserve the
window's total signal energy after perturbation (renormalizing back to the original per-channel
std) — this avoids pushing a z-scored window outside the range the model was trained on, which
would confound "the model reacts to more sigma power" with "the model reacts to an
out-of-distribution input." The per-channel filtering loop was originally a CPU-bound Python loop
over all channels, windows, and scale factors — the actual runtime bottleneck in subject-level
perturbation runs (not the GPU forward pass, despite that being the more obvious suspect) — and was
vectorized via `sosfiltfilt(..., axis=-1)` over the whole channel array at once, validated to
produce numerically identical output to the original loop.
