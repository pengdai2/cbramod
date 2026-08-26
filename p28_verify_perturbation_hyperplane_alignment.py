"""
p28_verify_perturbation_hyperplane_alignment.py

Verifies whether a band-power perturbation's feature-space displacement moves a subject's "virtual"
position toward the side of real bipolar/control space consistent with the model's own, actually
measured probability change -- the methodology worked out (through several wrong turns, recorded in
docs/sigma_band_causal_investigation.md Section 16.9-16.12) for checking this rigorously rather than
asserting it.

WHY NOT JUST DOT-PRODUCT THE DISPLACEMENT AGAINST A FITTED LOGISTIC/RIDGE MODEL (`w_model`)?
Because `w_model`'s coefficients are each a CONDITIONAL quantity ("this feature's contribution,
holding the others fixed"), and summing them across a displacement that moves every feature
simultaneously is only valid under exact linear additivity -- which a broadband control experiment
independently falsified (Section 16.4). This script instead uses a discriminant that never assumes
any feature is held fixed: distance to the real bipolar/control group centroids,
    disc(x) = ||(x - control_centroid)/sigma||  -  ||(x - bipolar_centroid)/sigma||
(sigma = per-feature std, so no single feature's raw scale dominates the distance). This is ALSO
provably a hyperplane -- expanding the squared-distance difference algebraically, the quadratic terms
cancel, leaving a linear function of x with normal vector proportional to
(bipolar_centroid - control_centroid)/sigma^2 -- but one whose weight on each feature is that
feature's own simple, MARGINAL difference between the groups, not a jointly-fit conditional one. See
Section 16.12 for why only the marginal version stays coherent for a perturbed (not just real) point.

WHAT THIS SCRIPT CHECKS, IN ORDER (each one caught a real bug or wrong assumption during development):
  1. Circularity: the reference population used to build the centroids must NOT contain any of the
     subjects being tested (subject_id overlap is checked and the overlapping subjects are dropped
     from the reference set automatically, with a printed warning if any were found).
  2. Metric validity: the discriminant is applied to the test subjects' own REAL, unperturbed
     positions and checked against their real ground truth -- confirms the metric captures real
     signal (via classification accuracy) before trusting it on perturbed positions.
  3. Displacement formula sanity: confirms fractions sum to 1 after perturbation (energy conserved).
  4. The actual check: for each band, each test subject's own baseline composition (from ALL their
     real windows, not a cohort average) is displaced by the analytic total-energy-preserving
     formula, and the resulting shift in discriminant score is compared, in SIGN, against the
     subject-grouped (not flat-pooled -- see the module-level NOTE below) mean of the real,
     measured per-window slope for that band.

NOTE on subject grouping: `actual_slope` is computed as mean(per-subject mean slope), not a flat
mean over all windows pooled together. These give IDENTICAL results only if every subject
contributed the same number of sampled windows to the perturbation run -- true for 4 of 5 bands in
the run this was built against, but NOT guaranteed in general (a flat pooled mean over-weights
whichever subjects happened to get more windows sampled). Always group by subject first.

Usage:
  python p28_verify_perturbation_hyperplane_alignment.py \
      --test-band-power-csv scratch/mlp-128-absolute_band_power_analysis.csv \
      --reference-csv analysis/absolute_band_power_analysis-bpd_ctl.csv \
      --perturbation-csv-template "scratch/mlp-128-{band}_band_power_perturbation.csv" \
      --bands delta,theta,alpha,sigma,beta \
      --scale-factor 1.5 \
      --output-csv p28_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]


def relpower_col(band: str) -> str:
    return f"{band}_real_relpower"


def displacement_vector(p_dict: dict, perturbed_band: str, k: float) -> dict:
    """Exact analytic change in every band's relative-power fraction under a total-energy-preserving
    perturbation of `perturbed_band` by scale factor k. See cbramod_common.perturb_window_band_power's
    docstring for the derivation; this is the same relationship expressed in fraction-of-total terms."""
    p_X = p_dict[perturbed_band]
    denom = k**2 * p_X + (1 - p_X)
    disp = {}
    for band, p_b in p_dict.items():
        new_frac = (k**2 * p_b / denom) if band == perturbed_band else (p_b / denom)
        disp[band] = new_frac - p_b
    return disp


def build_reference_centroids(reference_csv: Path, exclude_subject_ids: set, bands: list) -> dict:
    """Builds bipolar/control centroids and per-feature std from a reference population CSV,
    explicitly excluding any subject_id also present in the test set (checked and warned loudly,
    not silently) to avoid the circularity that motivated this whole check being built carefully."""
    cols = [relpower_col(b) for b in bands]
    df = pd.read_csv(reference_csv)
    subj = df.groupby("subject_id").agg({**{c: "mean" for c in cols}, "ground_truth": "first"})

    overlap = set(subj.index) & exclude_subject_ids
    if overlap:
        print(f"  [Warning] Reference population overlaps test set by {len(overlap)} subject(s) -- "
              f"dropping them from the reference to avoid circularity.")
        subj = subj.loc[~subj.index.isin(exclude_subject_ids)]
    else:
        print(f"  Reference population has zero overlap with the test set -- no circularity.")

    n_bipolar, n_control = (subj["ground_truth"] == 1).sum(), (subj["ground_truth"] == 0).sum()
    print(f"  Reference population (post-exclusion): n={len(subj)} (bipolar={n_bipolar}, control={n_control})")

    bipolar_centroid = subj[subj["ground_truth"] == 1][cols].mean().values
    control_centroid = subj[subj["ground_truth"] == 0][cols].mean().values
    group_std = subj[cols].std().values
    return {"bipolar_centroid": bipolar_centroid, "control_centroid": control_centroid, "group_std": group_std}


def disc_score(x: np.ndarray, centroids: dict) -> float:
    """Positive = closer to bipolar centroid; negative = closer to control centroid, in std-normalized
    (per-feature) Euclidean distance. Provably linear in x (see module docstring) despite being
    expressed as a distance difference."""
    control_dist = np.linalg.norm((x - centroids["control_centroid"]) / centroids["group_std"])
    bipolar_dist = np.linalg.norm((x - centroids["bipolar_centroid"]) / centroids["group_std"])
    return control_dist - bipolar_dist


def validate_metric_on_real_subjects(test_composition: pd.DataFrame, test_ground_truth: pd.Series,
                                      centroids: dict, bands: list) -> None:
    """Sanity check BEFORE trusting the metric on perturbed (virtual) positions: does it separate
    the test subjects' own REAL, unperturbed positions in the expected direction? These subjects were
    never used to build the centroids, so this is a real, non-circular validation."""
    cols = [relpower_col(b) for b in bands]
    disc_scores = test_composition[cols].apply(lambda row: disc_score(row.values, centroids), axis=1)
    bip_mean = disc_scores[test_ground_truth == 1].mean()
    ctl_mean = disc_scores[test_ground_truth == 0].mean()
    accuracy = ((disc_scores > 0) == (test_ground_truth == 1)).mean()
    print(f"  On real (unperturbed) test subjects: bipolar mean disc={bip_mean:+.3f}, "
          f"control mean disc={ctl_mean:+.3f}, sign(disc) classification accuracy={accuracy:.2f}")
    if bip_mean <= ctl_mean:
        print("  [Warning] Metric does NOT separate real subjects in the expected direction -- "
              "do not trust downstream perturbation-alignment results from this reference population.")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify perturbation displacement direction against real measured probability change."
    )
    parser.add_argument("--test-band-power-csv", type=str, required=True,
                         help="Per-window band-power CSV for the test cohort (has ground_truth, all *_real_relpower columns)")
    parser.add_argument("--reference-csv", type=str, required=True,
                         help="Larger, independent reference population CSV (has ground_truth) used to build group centroids")
    parser.add_argument("--perturbation-csv-template", type=str, required=True,
                         help="Path template with a {band} placeholder, e.g. 'scratch/mlp-128-{band}_band_power_perturbation.csv'")
    parser.add_argument("--bands", type=str, default=",".join(DEFAULT_BANDS),
                         help="Comma-separated bands to check (default: delta,theta,alpha,sigma,beta)")
    parser.add_argument("--scale-factor", type=float, default=1.5, help="Perturbation scale factor k")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save the per-band results table")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    bands = args.bands.split(",")

    test_df = pd.read_csv(args.test_band_power_csv)
    test_subject_ids = set(test_df["subject_id"].unique())
    test_ground_truth = test_df.groupby("subject_id")["ground_truth"].first()
    test_composition = test_df.groupby("subject_id")[[relpower_col(b) for b in bands]].mean()
    test_composition_norm = test_composition.div(test_composition.sum(axis=1), axis=0)
    assert np.allclose(test_composition_norm.sum(axis=1), 1.0), "subject compositions don't sum to 1 -- bug"

    print("Step 1-2: building reference centroids and validating the discriminant metric")
    centroids = build_reference_centroids(Path(args.reference_csv), test_subject_ids, bands)
    validate_metric_on_real_subjects(test_composition, test_ground_truth, centroids, bands)

    print(f"\nStep 3-4: per-band virtual-subject displacement vs. real measured slope (k={args.scale_factor})")
    results = []
    for band in bands:
        pert_path = Path(args.perturbation_csv_template.format(band=band))
        if not pert_path.exists():
            print(f"  [Warning] {pert_path} not found, skipping {band}")
            continue
        pert = pd.read_csv(pert_path)
        # Grouped by subject first, THEN averaged across subjects -- NOT a flat pooled mean (see
        # module docstring's NOTE: these differ whenever subjects contributed different window counts).
        actual_slope = pert.groupby("subject_id")["slope"].mean().mean()

        shifts = []
        for sid in test_composition_norm.index:
            p_dict = {b: test_composition_norm.loc[sid, relpower_col(b)] for b in bands}
            disp = displacement_vector(p_dict, band, args.scale_factor)
            x_before = np.array([p_dict[b] for b in bands])
            x_after = x_before + np.array([disp[b] for b in bands])
            assert abs(x_after.sum() - 1.0) < 1e-6, "displaced fractions don't sum to 1 -- bug in displacement_vector"
            shifts.append(disc_score(x_after, centroids) - disc_score(x_before, centroids))
        shifts = np.array(shifts)

        agree = np.sign(shifts.mean()) == np.sign(actual_slope)
        results.append({
            "band": band, "n_subjects": len(shifts), "mean_shift": shifts.mean(),
            "frac_shift_toward_bipolar": float((shifts > 0).mean()),
            "actual_slope": actual_slope, "agree": agree,
        })
        print(f"  {band:6s}: mean_shift={shifts.mean():+.4f}  frac_toward_bipolar={float((shifts>0).mean()):.2f}  "
              f"actual_slope={actual_slope:+.4f}  {'AGREE' if agree else 'DISAGREE -- see Section 16.9 before trusting w_model here'}")

    results_df = pd.DataFrame(results)
    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
