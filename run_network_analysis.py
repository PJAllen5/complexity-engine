#!/usr/bin/env python3
"""Run network-science analysis on graph-shaped (edge list) data."""

import argparse
from pathlib import Path

from complexity_engine.network_analysis import analyze_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze graph-structured data and compute network-science metrics.")
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parent / "data" / "network_example.csv"), help="Path to a CSV/Excel edge list, or a comma-separated list of files to merge (e.g. multiple session files sharing the same columns).")
    parser.add_argument("--source-col", type=str, default="source", help="Column giving one endpoint of each edge.")
    parser.add_argument("--target-col", type=str, default="target", help="Column giving the other endpoint of each edge.")
    parser.add_argument("--weight-col", type=str, default="weight", help="Optional edge weight column; pass an empty string to ignore weights.")
    parser.add_argument("--condition-col", type=str, default="condition", help="Optional column identifying which network each edge belongs to; pass an empty string for a single network.")
    parser.add_argument("--directed", action="store_true", help="Treat edges as directed. Changes several metrics' definitions -- see the report's note when this is set.")
    parser.add_argument("--compute-centrality", action="store_true", help="Also compute and report per-node degree/betweenness/eigenvector centrality.")
    parser.add_argument(
        "--compute-small-world",
        action="store_true",
        help="Also compute the small-world sigma/omega statistics (compares against randomized reference graphs). "
        "Only well-defined for an undirected, fully connected network with >= 4 nodes; NaN otherwise.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for the randomized reference graphs used by --compute-small-world.")
    parser.add_argument("--domain", type=str, default="network", help="Domain profile to apply.")
    parser.add_argument("--domain-config", type=str, default=None, help="Optional path to a YAML or JSON config file containing custom domain profiles.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "network"),
        help="Where to write the generated report and CSV outputs. Defaults to a subdirectory distinct from "
        "run_analysis.py's default output dir, since both write a report.md.",
    )
    args = parser.parse_args()

    data_files = [item.strip() for item in args.data.split(",") if item.strip()]
    data_path = data_files[0] if len(data_files) == 1 else data_files

    result = analyze_network(
        data_path=data_path,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col or None,
        condition_col=args.condition_col or None,
        directed=args.directed,
        compute_centrality=args.compute_centrality,
        compute_small_world=args.compute_small_world,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        domain=args.domain,
        domain_config=args.domain_config,
    )

    if result["missing_columns"]:
        print(f"Warning: dataset is missing columns expected by the '{args.domain}' domain profile: {', '.join(result['missing_columns'])}")
    provenance = result["provenance"]
    if provenance["rows_dropped_missing_values"]:
        print(f"Warning: dropped {provenance['rows_dropped_missing_values']} row(s) missing a required value.")
    if provenance["duplicate_rows_detected"]:
        print(f"Warning: detected {provenance['duplicate_rows_detected']} exact duplicate row(s) (not removed automatically; see report.md).")
    print(f"Saved network metrics to: {result['metrics_csv']}")
    if result["centrality_csv"]:
        print(f"Saved node centrality to: {result['centrality_csv']}")
    print(f"Saved report to: {result['report_md']}")
    print(f"Saved self-contained HTML report to: {result['report_html']}")


if __name__ == "__main__":
    main()
