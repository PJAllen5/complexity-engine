"""Command-line entry points for both pipelines.

Installed as the `complexity-analysis` (time series) and `complexity-network` commands;
run_analysis.py and run_network_analysis.py in the repository call the same functions.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from complexity_engine.analysis import analyze_dataset
from complexity_engine.network_analysis import analyze_network

# Running from a checkout of the repository (rather than an installed copy) keeps the
# original defaults: the bundled example data, and results in the repository's outputs/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_IN_CHECKOUT = (_REPO_ROOT / "run_analysis.py").exists()


def _example(name: str) -> str | None:
    path = _REPO_ROOT / "data" / name
    return str(path) if _IN_CHECKOUT and path.exists() else None


def _output_root() -> Path:
    return _REPO_ROOT / "outputs" if _IN_CHECKOUT else Path.cwd() / "outputs"


def timeseries_main(argv: Sequence[str] | None = None) -> None:
    """`complexity-analysis`: the time-series pipeline (same as run_analysis.py)."""
    parser = argparse.ArgumentParser(description="Analyze scientific time-series and compute complexity metrics.")
    parser.add_argument("--data", type=str, default=_example("physio_example.csv"), required=_example("physio_example.csv") is None, help="Path to the CSV/Excel dataset, or a comma-separated list of files to merge (e.g. multiple session files sharing the same columns).")
    parser.add_argument("--signal-cols", type=str, default=None, help="Comma-separated list of numeric signal columns to analyze.")
    parser.add_argument("--condition-col", type=str, default="condition", help="Column that identifies the condition or group.")
    parser.add_argument("--unit-cols", type=str, default=None, help="Comma-separated columns identifying one independent recording (e.g. subject_id,trial); complexity metrics are computed within each recording, then averaged per condition. Defaults to auto-detecting subject_id/subject/participant/session/trial. Pass 'none' to treat each condition as a single recording.")
    parser.add_argument("--domain", type=str, default="physiology", help="Scientific domain profile to apply: physiology, cognitive, stress, or another configured profile.")
    parser.add_argument("--domain-config", type=str, default=None, help="Optional path to a YAML or JSON config file containing custom domain profiles.")
    parser.add_argument("--output-dir", type=str, default=str(_output_root()), help="Where to write the generated report and CSV outputs.")
    args = parser.parse_args(argv)

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


def network_main(argv: Sequence[str] | None = None) -> None:
    """`complexity-network`: the network pipeline (same as run_network_analysis.py)."""
    parser = argparse.ArgumentParser(description="Analyze graph-structured data and compute network-science metrics.")
    parser.add_argument("--data", type=str, default=_example("network_example.csv"), required=_example("network_example.csv") is None, help="Path to a CSV/Excel edge list, or a comma-separated list of files to merge (e.g. multiple session files sharing the same columns).")
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
    parser.add_argument("--compute-communities", action="store_true", help="Also write each node's detected community (the partition modularity scores) to node_communities.csv.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for the randomized reference graphs used by --compute-small-world.")
    parser.add_argument("--domain", type=str, default="network", help="Domain profile to apply.")
    parser.add_argument("--domain-config", type=str, default=None, help="Optional path to a YAML or JSON config file containing custom domain profiles.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_output_root() / "network"),
        help="Where to write the generated report and CSV outputs. Defaults to a subdirectory distinct from "
        "run_analysis.py's default output dir, since both write a report.md.",
    )
    args = parser.parse_args(argv)

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
        compute_communities=args.compute_communities,
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
    if result["communities_csv"]:
        print(f"Saved node communities to: {result['communities_csv']}")
    print(f"Saved report to: {result['report_md']}")
    print(f"Saved self-contained HTML report to: {result['report_html']}")
