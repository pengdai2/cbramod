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
