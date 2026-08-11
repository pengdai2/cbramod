"""
Post-Mortem Subject Gatekeeper & Training Manifest Generator.

Parses all subject metadata JSON files, applies quality acceptance rules, 
and outputs a training manifest pointing only to valid windows from accepted subjects.
"""

import json
import argparse
from pathlib import Path
import pandas as pd

from cbramod_utils import evaluate_subject_quality


def evaluate_subject_gatekeeping(
    json_path: Path, 
    max_subj_rejection_rate: float = 0.20,
    max_n3_rejection_rate: float = 0.15,
    min_valid_sleep_hours: float = 4.0,
    window_duration_sec: float = 30.0
) -> dict:
    """Evaluates a single subject JSON against acceptance rules."""
    with open(json_path, 'r') as f:
        meta = json.load(f)

    subject_id = meta.get("subject_id", json_path.stem)

    _, qa_metrics, _ = evaluate_subject_quality(meta, 
                                                max_subj_rejection_rate=max_subj_rejection_rate,
                                                max_n3_rejection_rate=max_n3_rejection_rate,
                                                min_valid_sleep_hours=min_valid_sleep_hours,
                                                window_duration_sec=window_duration_sec)
 
    return {
        "subject_id": subject_id,
        "json_path": str(json_path),
        **qa_metrics
    }


def generate_manifest(meta_dir: Path, output_dir: Path):
    """Scans all JSON files and outputs a cohort gatekeeping report."""
    results = []
    json_files = list(meta_dir.glob("*.json"))

    for jf in json_files:
        res = evaluate_subject_gatekeeping(jf)
        results.append(res)

    report_df = pd.DataFrame(results)
    output_csv = output_dir / "cohort_manifest.csv"
    report_df.to_csv(output_csv, index=False)

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
