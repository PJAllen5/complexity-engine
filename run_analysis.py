#!/usr/bin/env python3
"""Run the physiological/cognitive complexity analysis MVP."""

import argparse
from pathlib import Path

from complexity_engine.analysis import analyze_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scientific time-series and compute complexity metrics.")
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parent / "data" / "physio_example.csv"), help="Path to the CSV/Excel dataset.")
    parser.add_argument("--signal-cols", type=str, default=None, help="Comma-separated list of numeric signal columns to analyze.")
    parser.add_argument("--condition-col", type=str, default="condition", help="Column that identifies the condition or group.")
    parser.add_argument("--domain", type=str, default="physiology", help="Scientific domain profile to apply: physiology, cognitive, stress, or another configured profile.")
    parser.add_argument("--domain-config", type=str, default=None, help="Optional path to a YAML or JSON config file containing custom domain profiles.")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"), help="Where to write the generated report and CSV outputs.")
    args = parser.parse_args()

    signal_cols = None
    if args.signal_cols:
        signal_cols = [item.strip() for item in args.signal_cols.split(",") if item.strip()]

    result = analyze_dataset(
        data_path=args.data,
        signal_cols=signal_cols,
        condition_col=args.condition_col,
        output_dir=args.output_dir,
        domain=args.domain,
        domain_config=args.domain_config,
    )

    print(f"Selected signal columns: {', '.join(result['selected_signals'])}")
    print(f"Saved summary to: {result['summary_csv']}")
    print(f"Saved complexity metrics to: {result['complexity_csv']}")
    print(f"Saved report to: {result['report_md']}")


if __name__ == "__main__":
    main()
