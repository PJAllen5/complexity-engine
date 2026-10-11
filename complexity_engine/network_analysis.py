"""Analysis workflow for graph-shaped (edge list) data.

Parallels analysis.py's time-series pipeline but is kept as a separate entry point
rather than folded into analyze_dataset(): the two operate on genuinely different
data shapes (rows-are-observations time series vs. rows-are-edges graphs), so
sharing one function would mean branching on data shape throughout, instead of two
small, narrow pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .analysis import _as_code_block, _collapse_blank_lines, _describe_data_source, clean_dataset, load_dataset
from .domain_profiles import (
    DomainProfile,
    get_domain_profile,
    load_domain_profiles,
    validate_metric_parameters,
    validate_recommended_metrics,
)
from .network_metrics import (
    NETWORK_METRIC_KEYS,
    build_graph,
    compute_network_metrics,
    compute_node_centrality,
    compute_small_world_metrics,
)
from .plots import plot_degree_distribution
from .report_export import export_html_report

_SMALL_WORLD_METRICS = ("small_world_sigma", "small_world_omega")

_DEFAULT_METRIC_EXPLANATIONS = {
    "avg_clustering": "Average clustering measures how often a node's neighbors are also connected to each other.",
    "transitivity": "Transitivity is the global fraction of connected triples that form a triangle.",
    "modularity": "Modularity measures how cleanly the network splits into separate communities.",
    "avg_shortest_path_length": "Average shortest path length is the mean hop count between reachable node pairs.",
    "density": "Density is the fraction of all possible edges that are actually present.",
    "mean_degree": "Mean degree is the average number of connections per node.",
    "mean_in_degree": "Mean in-degree is the average number of incoming connections per node.",
    "mean_out_degree": "Mean out-degree is the average number of outgoing connections per node.",
}


def load_edge_list(data_path: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Load one edge-list file, or merge several from the same multi-session study.

    Delegates to analysis.py's load_dataset(), which is data-shape-agnostic (generic
    CSV/Excel loading, multi-file merge, and column-mismatch validation -- nothing
    about it is specific to time series), rather than duplicating that logic here.
    """
    return load_dataset(data_path)


def _missing_expected_columns(df: pd.DataFrame, domain_profile: DomainProfile) -> list[str]:
    return [col for col in domain_profile.expected_columns if col not in df.columns]


def _generate_network_interpretation_lines(metrics_df: pd.DataFrame, domain_profile: DomainProfile) -> list[str]:
    lines = ["## Interpretation", ""]
    if domain_profile.analysis_question:
        lines.append(f"Guiding question: {domain_profile.analysis_question}")
        lines.append("")

    metrics_of_interest = list(domain_profile.recommended_metrics) or list(_DEFAULT_METRIC_EXPLANATIONS)
    for metric in metrics_of_interest:
        explanation = domain_profile.interpretation_rules.get(metric) or _DEFAULT_METRIC_EXPLANATIONS.get(metric)
        if explanation:
            lines.append(f"- {metric}: {explanation}")
    lines.append("")

    if metrics_df.shape[0] < 2:
        return lines

    sorted_rows = metrics_df.sort_values("condition")
    first, second = sorted_rows.iloc[0], sorted_rows.iloc[-1]
    for metric in metrics_of_interest:
        if metric not in sorted_rows.columns:
            continue
        first_val, second_val = float(first[metric]), float(second[metric])
        if pd.isna(first_val) or pd.isna(second_val) or first_val == second_val:
            continue
        direction = "higher" if second_val > first_val else "lower"
        rule = domain_profile.interpretation_rules.get(metric, "")
        metric_label = metric.replace("_", " ")
        lines.append(
            f"- {metric_label} is {direction} in {second['condition']} than in {first['condition']}. {rule}".rstrip()
        )

    small_world_metrics = [m for m in _SMALL_WORLD_METRICS if m in metrics_df.columns]
    if small_world_metrics:
        lines.append("")
        lines.append("### Small-world statistics")
        for metric in small_world_metrics:
            explanation = domain_profile.interpretation_rules.get(metric)
            if explanation:
                lines.append(f"- {metric}: {explanation}")
        for metric in small_world_metrics:
            first_val, second_val = float(first[metric]), float(second[metric])
            if pd.isna(first_val) or pd.isna(second_val) or first_val == second_val:
                continue
            direction = "higher" if second_val > first_val else "lower"
            metric_label = metric.replace("_", " ")
            lines.append(
                f"- {metric_label} is {direction} in {second['condition']} than in {first['condition']}."
            )
    return lines


def analyze_network(
    data_path: str | Path | Sequence[str | Path],
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = "weight",
    condition_col: str | None = "condition",
    directed: bool = False,
    compute_centrality: bool = False,
    compute_small_world: bool = False,
    random_seed: int = 42,
    output_dir: str | Path = "outputs",
    domain: str | None = "network",
    domain_config: str | Path | None = None,
) -> dict[str, pd.DataFrame | dict | str]:
    load_domain_profiles(domain_config)
    domain_profile = get_domain_profile(domain)
    validate_recommended_metrics(domain_profile, NETWORK_METRIC_KEYS)
    # Network metrics take no parameters; this rejects any rather than silently ignoring them.
    validate_metric_parameters(domain_profile, {})
    df = load_edge_list(data_path)
    missing_columns = _missing_expected_columns(df, domain_profile)
    has_condition = condition_col is not None and condition_col in df.columns
    required_cols = [source_col, target_col] + ([condition_col] if has_condition else [])
    df, provenance = clean_dataset(df, required_cols=required_cols)

    if has_condition:
        conditions = sorted(df[condition_col].dropna().unique(), key=str)
        groups = {condition: df[df[condition_col] == condition] for condition in conditions}
    else:
        groups = {"all": df}

    graphs = {
        condition: build_graph(edges, source_col=source_col, target_col=target_col, weight_col=weight_col, directed=directed)
        for condition, edges in groups.items()
    }
    metrics_df = pd.DataFrame.from_records(
        [
            {
                "condition": condition,
                **compute_network_metrics(graph),
                **(compute_small_world_metrics(graph, seed=random_seed) if compute_small_world else {}),
            }
            for condition, graph in graphs.items()
        ]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "network_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    centrality_df = pd.DataFrame()
    centrality_path = None
    if compute_centrality:
        centrality_frames = []
        for condition, graph in graphs.items():
            frame = compute_node_centrality(graph)
            frame.insert(0, "condition", condition)
            centrality_frames.append(frame)
        centrality_df = pd.concat(centrality_frames, ignore_index=True) if centrality_frames else pd.DataFrame()
        centrality_path = output_dir / "node_centrality.csv"
        centrality_df.to_csv(centrality_path, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    degree_plot_path = plot_degree_distribution(graphs, plots_dir / "degree_distribution.png")

    report_path = output_dir / "report.md"
    report_lines = [
        "# Network Analysis Report",
        "",
        f"- Domain: {domain_profile.name}",
        f"- Domain description: {domain_profile.description}",
        f"- Analysis question: {domain_profile.analysis_question}" if domain_profile.analysis_question else "",
        f"- Data source: {_describe_data_source(data_path)}",
        f"- Edge columns: {source_col}, {target_col}" + (f", {weight_col}" if weight_col in df.columns else ""),
        f"- Condition column: {condition_col if has_condition else 'none (single network)'}",
        f"- Directed: {directed}",
        "",
    ]
    if domain_profile.signal_roles:
        report_lines.append("## Domain edge roles")
        for role, description in domain_profile.signal_roles.items():
            report_lines.append(f"- {role}: {description}")
        report_lines.append("")
    if missing_columns:
        report_lines.append(
            f"Note: the '{domain_profile.name}' domain profile expects columns "
            f"[{', '.join(domain_profile.expected_columns)}]; missing from this dataset: {', '.join(missing_columns)}."
        )
        report_lines.append("")
    report_lines.append("## Data provenance")
    report_lines.append(f"- Input rows: {provenance['input_rows']}")
    report_lines.append(f"- Duplicate rows detected (not removed automatically): {provenance['duplicate_rows_detected']}")
    report_lines.append(
        f"- Rows dropped for missing values in {provenance['missing_value_columns_checked']}: "
        f"{provenance['rows_dropped_missing_values']}"
    )
    report_lines.append(f"- Rows analyzed: {provenance['output_rows']}")
    report_lines.append("")
    if directed:
        report_lines.append(
            "Note: this is a directed network. n_connected_components counts weakly connected "
            "components (ignoring direction); avg_shortest_path_length is computed on the "
            "largest strongly connected component (n_strongly_connected_components), the only "
            "subgraph where every node pair is mutually reachable."
        )
        report_lines.append("")
    if compute_small_world:
        report_lines.append(
            "Note: small_world_sigma/small_world_omega are NaN for any condition whose network "
            "isn't undirected, fully connected, and at least 4 nodes/2 edges -- these statistics "
            "compare against randomized reference graphs and aren't well-defined otherwise."
        )
        report_lines.append("")
    report_lines.append("## Network metrics")
    report_lines.extend(_as_code_block(metrics_df.to_string(index=False)))
    if compute_centrality and not centrality_df.empty:
        report_lines.append("## Node centrality")
        report_lines.extend(_as_code_block(centrality_df.to_string(index=False)))
    if degree_plot_path:
        report_lines.extend(["## Visualizations", f"![degree_distribution](plots/{Path(degree_plot_path).name})", ""])
    report_lines.extend(_generate_network_interpretation_lines(metrics_df, domain_profile))
    report_lines = _collapse_blank_lines(report_lines)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    html_path = export_html_report(report_path)

    return {
        "data": df,
        "graphs": graphs,
        "domain_profile": domain_profile.as_dict(),
        "missing_columns": missing_columns,
        "provenance": provenance,
        "network_metrics_summary": metrics_df,
        "node_centrality": centrality_df,
        "metrics_csv": str(metrics_path),
        "centrality_csv": str(centrality_path) if centrality_path else None,
        "degree_plot_path": degree_plot_path,
        "report_md": str(report_path),
        "report_html": html_path,
    }
