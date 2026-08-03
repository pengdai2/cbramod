"""
Post-Mortem Subject Gatekeeper & Training Manifest Generator.

Parses all subject metadata JSON files, applies quality acceptance rules, 
and outputs a training manifest pointing only to valid windows from accepted subjects.
"""

import json
import argparse
from pathlib import Path
import pandas as pd

def evaluate_subject_gatekeeping(
    json_path: Path, 
    max_subj_rejection_rate: float = 0.15,
    max_n3_rejection_rate: float = 0.10,
    min_valid_sleep_hours: float = 4.5
) -> dict:
    """Evaluates a single subject JSON against acceptance rules."""
    with open(json_path, 'r') as f:
        meta = json.load(f)

    subject_id = meta.get("subject_id", json_path.stem)
    slices = meta.get("slices", [])
    stages = meta.get("stages", [])

    total_wins = len(slices)
    if total_wins == 0:
        return {"subject_id": subject_id, "accepted": False, "reason": "EMPTY_SLICES"}

    df = pd.DataFrame(slices)
    df['stage'] = stages[:len(df)]

    invalid_wins = (~df['is_valid']).sum()
    overall_rejection_rate = invalid_wins / total_wins
    valid_duration_hrs = (df['is_valid'].sum() * 30.0) / 3600.0

    # Check N3 specific rejection rate
    n3_df = df[df['stage'].str.upper() == 'N3']
    n3_rejection_rate = (~n3_df['is_valid']).sum() / len(n3_df) if len(n3_df) > 0 else 0.0

    # Policy Checks
    reasons = []
    if overall_rejection_rate > max_subj_rejection_rate:
        reasons.append(f"HIGH_REJECTION_RATE ({overall_rejection_rate:.1%})")
    
    if n3_rejection_rate > max_n3_rejection_rate:
        reasons.append(f"HIGH_N3_REJECTION ({n3_rejection_rate:.1%})")
        
    if valid_duration_hrs < min_valid_sleep_hours:
        reasons.append(f"SHORT_VALID_SLEEP ({valid_duration_hrs:.2f} hrs)")

    accepted = len(reasons) == 0

    return {
        "subject_id": subject_id,
        "json_path": str(json_path),
        "total_windows": total_wins,
        "valid_windows": df['is_valid'].sum(),
        "overall_rejection_rate": overall_rejection_rate,
        "n3_rejection_rate": n3_rejection_rate,
        "valid_hours": valid_duration_hrs,
        "accepted": accepted,
        "reasons": "; ".join(reasons) if not accepted else "PASSED"
    }


def generate_manifest(meta_dir: Path, output_dir: Path):
    """Scans all JSON files and outputs a cohort gatekeeping report."""
    results = []
    json_files = list(meta_dir.glob("*.json"))

    for jf in json_files:
        res = evaluate_subject_gatekeeping(jf)
        results.append(res)

    report_df = pd.DataFrame(results)
    report_df.to_csv(output_dir / "subject_manifest.csv, index=False)

    total_subjs = len(report_df)
    accepted_subjs = report_df['accepted'].sum()
    
    print("=" * 60)
    print(f" COHORT GATEKEEPING SUMMARY")
    print("=" * 60)
    print(f" Total Subjects Evaluated: {total_subjs}")
    print(f" Accepted Subjects:        {accepted_subjs} ({(accepted_subjs/total_subjs)*100:.1f}%)")
    print(f" Rejected Subjects:        {total_subjs - accepted_subjs} ({((total_subjs - accepted_subjs)/total_subjs)*100:.1f}%)")
    print(f" Cohort Report Saved To:   {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Screening and Subject Gatekeeping")
    parser.add_argument("--sliced_dir", type=str, required=True, help="Directory containing sliced .npy and _meta.json files")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for CSV manifests")

    args = parser.parse_args()
    generate_manifest(Path(args.sliced_dir), Path(args.output_dir))
