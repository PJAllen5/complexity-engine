#!/usr/bin/env python3
"""Run the physiological/cognitive complexity analysis MVP."""

import argparse
from pathlib import Path

from complexity_engine.analysis import analyze_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scientific time-series and compute complexity metrics.")
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parent / "data" / "physio_example.csv"), help="Path to the CSV/Excel dataset, or a comma-separated list of files to merge (e.g. multiple session files sharing the same columns).")
    parser.add_argument("--signal-cols", type=str, default=None, help="Comma-separated list of numeric signal columns to analyze.")
    parser.add_argument("--condition-col", type=str, default="condition", help="Column that identifies the condition or group.")
    parser.add_argument("--unit-cols", type=str, default=None, help="Comma-separated columns identifying one independent recording (e.g. subject_id,trial); complexity metrics are computed within each recording, then averaged per condition. Defaults to auto-detecting subject_id/subject/participant/session/trial. Pass 'none' to treat each condition as a single recording.")
    parser.add_argument("--domain", type=str, default="physiology", help="Scientific domain profile to apply: physiology, cognitive, stress, or another configured profile.")
    parser.add_argument("--domain-config", type=str, default=None, help="Optional path to a YAML or JSON config file containing custom domain profiles.")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"), help="Where to write the generated report and CSV outputs.")
    args = parser.parse_args()

    signal_cols = None
    if args.signal_cols:
        signal_cols = [item.strip() for item in args.signal_cols.split(",") if item.strip()]

    unit_cols = None
    if args.unit_cols is not None:
        unit_cols = [] if args.unit_cols.strip().lower() == "none" else [item.strip() for item in args.unit_cols.split(",") if item.strip()]

    data_files = [item.strip() for item in args.data.split(",") if item.strip()]
    data_path = data_files[0] if len(data_files) == 1 else data_files

    result = analyze_dataset(
        data_path=data_path,
        signal_cols=signal_cols,
        condition_col=args.condition_col,
        output_dir=args.output_dir,
        domain=args.domain,
        domain_config=args.domain_config,
        unit_cols=unit_cols,
    )

    print(f"Selected signal columns: {', '.join(result['selected_signals'])}")
    print(f"Recording unit columns: {', '.join(result['unit_columns']) or 'none'}")
    if result["missing_columns"]:
        print(f"Warning: dataset is missing columns expected by the '{args.domain}' domain profile: {', '.join(result['missing_columns'])}")
    provenance = result["provenance"]
    if provenance["rows_dropped_missing_values"]:
        print(f"Warning: dropped {provenance['rows_dropped_missing_values']} row(s) missing a required value.")
    if provenance["duplicate_rows_detected"]:
        print(f"Warning: detected {provenance['duplicate_rows_detected']} exact duplicate row(s) (not removed automatically; see report.md).")
    print(f"Saved summary to: {result['summary_csv']}")
    print(f"Saved complexity metrics to: {result['complexity_csv']}")
    print(f"Saved report to: {result['report_md']}")
    print(f"Saved self-contained HTML report to: {result['report_html']}")


if __name__ == "__main__":
    main()
