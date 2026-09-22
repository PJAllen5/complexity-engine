"""Analysis workflow for graph-shaped (edge list) data.

Parallels analysis.py's time-series pipeline but is kept as a separate entry point
rather than folded into analyze_dataset(): the two operate on genuinely different
data shapes (rows-are-observations time series vs. rows-are-edges graphs), so
sharing one function would mean branching on data shape throughout, instead of two
small, narrow pipelines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .domain_profiles import DomainProfile, get_domain_profile, load_domain_profiles
from .network_metrics import build_graph, compute_network_metrics, compute_node_centrality
from .plots import plot_degree_distribution

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


def load_edge_list(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


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
    return lines


def analyze_network(
    data_path: str | Path,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = "weight",
    condition_col: str | None = "condition",
    directed: bool = False,
    compute_centrality: bool = False,
    output_dir: str | Path = "outputs",
    domain: str | None = "network",
    domain_config: str | Path | None = None,
) -> dict[str, pd.DataFrame | dict | str]:
    load_domain_profiles(domain_config)
    domain_profile = get_domain_profile(domain)
    df = load_edge_list(data_path)
    missing_columns = _missing_expected_columns(df, domain_profile)

    has_condition = condition_col is not None and condition_col in df.columns
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
        [{"condition": condition, **compute_network_metrics(graph)} for condition, graph in graphs.items()]
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
        f"- Data source: {Path(data_path).name}",
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
    if directed:
        report_lines.append(
            "Note: this is a directed network. n_connected_components counts weakly connected "
            "components (ignoring direction); avg_shortest_path_length is computed on the "
            "largest strongly connected component (n_strongly_connected_components), the only "
            "subgraph where every node pair is mutually reachable."
        )
        report_lines.append("")
    report_lines.extend(["## Network metrics", metrics_df.to_string(index=False), ""])
    if compute_centrality and not centrality_df.empty:
        report_lines.extend(["## Node centrality", centrality_df.to_string(index=False), ""])
    if degree_plot_path:
        report_lines.extend(["## Visualizations", f"![degree_distribution](plots/{Path(degree_plot_path).name})", ""])
    report_lines.extend(_generate_network_interpretation_lines(metrics_df, domain_profile))
    report_lines = [line for line in report_lines if line]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "data": df,
        "graphs": graphs,
        "domain_profile": domain_profile.as_dict(),
        "missing_columns": missing_columns,
        "network_metrics_summary": metrics_df,
        "node_centrality": centrality_df,
        "metrics_csv": str(metrics_path),
        "centrality_csv": str(centrality_path) if centrality_path else None,
        "degree_plot_path": degree_plot_path,
        "report_md": str(report_path),
    }
