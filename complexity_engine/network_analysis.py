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

import networkx as nx
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
    ATTRIBUTE_MIXING_COLUMNS,
    MAX_GREEDY_COMMUNITY_NODES,
    MAX_SMALL_WORLD_NODES,
    NETWORK_METRIC_KEYS,
    attribute_mixing,
    build_graph,
    compute_network_metrics,
    compute_node_centrality,
    compute_node_communities,
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


def load_node_table(
    data_path: str | Path | Sequence[str | Path],
    node_col: str = "node",
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Load a node table: one row per node, with a node id column and any attribute columns.

    The id column is renamed to "node". When condition_col is given and present, rows give
    each condition's own nodes and attributes. Rows without a node id are dropped. A node
    listed twice (twice in one condition, with a condition column) is an error, since its
    attributes would be ambiguous.
    """
    table = load_dataset(data_path)
    if node_col not in table.columns:
        raise ValueError(
            f"Node table has no '{node_col}' column (columns: {', '.join(map(str, table.columns))}); "
            "set the node id column with --node-col / node_col=."
        )
    table = table.dropna(subset=[node_col]).rename(columns={node_col: "node"})
    keys = [condition_col, "node"] if condition_col and condition_col in table.columns else ["node"]
    duplicated = table[table.duplicated(keys, keep=False)]
    if not duplicated.empty:
        examples = ", ".join(map(str, duplicated["node"].drop_duplicates().head(5)))
        scope = f" within a {condition_col}" if len(keys) == 2 else ""
        raise ValueError(f"Node table lists some nodes more than once{scope}: {examples}.")
    return table.reset_index(drop=True)


def _check_node_id_types(table_nodes: set, edge_nodes: set) -> None:
    """Raise when the node table and the edges name the same nodes with different types
    (e.g. 7 vs "7"), which would otherwise add every node as an isolate."""
    if not table_nodes or not edge_nodes or table_nodes & edge_nodes:
        return
    if {str(n) for n in table_nodes} & {str(n) for n in edge_nodes}:
        raise ValueError(
            "The node table and the edge list name nodes the same way but with different types "
            "(e.g. 7 vs \"7\"), so no node matches. Make the id columns the same type."
        )


def attach_node_table(
    graphs: dict, node_table: pd.DataFrame, condition_col: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Add the node table to each condition's graph, in place.

    Nodes in the table with no edges are added as isolates; their attributes, and those of
    every listed node, are set as node attributes (missing values are left unset). Edge
    endpoints the table doesn't list are kept, without attributes. Without a condition column
    in the table, it applies to every condition's graph.

    Returns one row per condition (n_table_nodes, isolates_added, nodes_not_in_table), and
    the number of table rows for conditions that have no edges (which are ignored).
    """
    per_condition = condition_col is not None and condition_col in node_table.columns
    attributes = [c for c in node_table.columns if c not in ("node", condition_col if per_condition else None)]
    _check_node_id_types(set(node_table["node"]), {n for g in graphs.values() for n in g.nodes})
    records = []
    for condition, graph in graphs.items():
        rows = node_table[node_table[condition_col] == condition] if per_condition else node_table
        listed = set(rows["node"])
        isolates = listed - set(graph.nodes)
        not_listed = set(graph.nodes) - listed
        graph.add_nodes_from(sorted(isolates, key=str))
        for attribute in attributes:
            values = rows.set_index("node")[attribute].dropna()
            nx.set_node_attributes(graph, values.to_dict(), name=attribute)
        records.append({
            "condition": condition,
            "n_table_nodes": len(listed),
            "isolates_added": len(isolates),
            "nodes_not_in_table": len(not_listed),
        })
    unmatched_rows = (
        int((~node_table[condition_col].isin(list(graphs))).sum()) if per_condition else 0
    )
    return pd.DataFrame.from_records(records), unmatched_rows


def _with_node_attributes(frame: pd.DataFrame, node_table: pd.DataFrame | None, condition_col: str | None) -> pd.DataFrame:
    """Per-node output with the node table's attribute columns joined on (condition and) node."""
    if node_table is None or frame.empty:
        return frame
    if condition_col is not None and condition_col in node_table.columns:
        table = node_table.rename(columns={condition_col: "condition"})
        return frame.merge(table, on=["condition", "node"], how="left")
    return frame.merge(node_table, on="node", how="left")


def _node_table_lines(
    stats: pd.DataFrame, unmatched_rows: int, attributes: Sequence[str], source: str
) -> list[str]:
    lines = ["## Nodes", f"Node table: {source}. Attributes: {', '.join(attributes) or 'none'}.", ""]
    for _, row in stats.iterrows():
        lines.append(
            f"- {row['condition']}: {row['n_table_nodes']} node(s) listed; {row['isolates_added']} with no ties "
            f"added as isolates; {row['nodes_not_in_table']} edge endpoint(s) not in the table, kept without "
            "attributes."
        )
    if unmatched_rows:
        lines.append(f"- {unmatched_rows} node-table row(s) name a condition with no edges, so they were ignored.")
    if stats["isolates_added"].sum():
        lines.append(
            "Isolates count toward n_nodes, density and mean_degree, and each one is its own connected component."
        )
    return [*lines, ""]


def _attribute_kind(values: pd.Series, declared: str = "") -> str:
    """The profile's declared kind, else numeric for a numeric column with more than two
    distinct values, else categorical (so 0/1 codes and booleans count as groups)."""
    if declared:
        return declared
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        return "categorical"
    return "numeric" if values.dropna().nunique() > 2 else "categorical"


def _attribute_mixing_lines(mixing: pd.DataFrame, domain_profile: DomainProfile, permutations: int, path: Path) -> list[str]:
    lines = [
        "## Attribute mixing",
        "How ties mix across node attributes, over nodes with a value and ties between two of them "
        "(self-loops excluded). assortativity is Newman's (2003) coefficient: 1 when every tie stays "
        "within a group, 0 for random mixing, negative when ties prefer other groups. same_group_tie_fraction "
        "is the share of ties within a group, and expected_same_group_fraction what random mixing with the "
        "same group sizes and degrees would give (McPherson et al., 2001). p_value is a two-sided "
        f"permutation test of assortativity ({permutations} shuffles of the attribute across nodes), and "
        "community_nmi the agreement between the attribute and the detected communities (Danon et al., 2005). "
        f"Values are in {path.name}.",
        "",
    ]
    for attribute, spec in domain_profile.node_attributes.items():
        if spec.get("role"):
            lines.append(f"- {attribute}: {spec['role']}")
    for metric in ("assortativity", "same_group_tie_fraction", "community_nmi"):
        if domain_profile.interpretation_rules.get(metric):
            lines.append(f"- {metric}: {domain_profile.interpretation_rules[metric]}")
    lines.append("")
    lines.extend(_as_code_block(mixing.to_string(index=False)))
    return lines


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
    compute_communities: bool = False,
    random_seed: int = 42,
    output_dir: str | Path = "outputs",
    domain: str | None = "network",
    domain_config: str | Path | None = None,
    node_table: str | Path | Sequence[str | Path] | None = None,
    node_col: str = "node",
    attributes: Sequence[str] | None = None,
    permutations: int = 1000,
    path_sources: int = 50,
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
    nodes_df = None
    node_stats = pd.DataFrame()
    unmatched_node_rows = 0
    if node_table is not None:
        nodes_df = load_node_table(node_table, node_col=node_col, condition_col=condition_col if has_condition else None)
        node_stats, unmatched_node_rows = attach_node_table(graphs, nodes_df, condition_col if has_condition else None)
    node_condition_col = condition_col if has_condition else None
    mixing_attributes = list(attributes) if attributes is not None else list(domain_profile.node_attributes)
    if mixing_attributes and nodes_df is None:
        raise ValueError(
            f"Attribute mixing for {', '.join(mixing_attributes)} needs a node table (--nodes / node_table=) "
            "giving each node's attributes."
        )
    if nodes_df is not None:
        missing = [a for a in mixing_attributes if a not in nodes_df.columns]
        if missing:
            raise ValueError(
                f"Node table has no column(s) {', '.join(missing)} for attribute mixing; "
                f"its columns are {', '.join(map(str, nodes_df.columns))}."
            )
    if permutations < 1:
        raise ValueError(f"permutations must be at least 1, got {permutations}.")
    metrics_df = pd.DataFrame.from_records(
        [
            {
                "condition": condition,
                **compute_network_metrics(graph, path_sources=path_sources, seed=random_seed),
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
        centrality_df = _with_node_attributes(centrality_df, nodes_df, node_condition_col)
        centrality_path = output_dir / "node_centrality.csv"
        centrality_df.to_csv(centrality_path, index=False)

    communities_df = pd.DataFrame()
    communities_path = None
    if compute_communities:
        community_frames = []
        for condition, graph in graphs.items():
            frame = compute_node_communities(graph, seed=random_seed)
            frame.insert(0, "condition", condition)
            community_frames.append(frame)
        communities_df = pd.concat(community_frames, ignore_index=True) if community_frames else pd.DataFrame()
        communities_df = _with_node_attributes(communities_df, nodes_df, node_condition_col)
        communities_path = output_dir / "node_communities.csv"
        communities_df.to_csv(communities_path, index=False)

    mixing_df = pd.DataFrame(columns=["condition", *ATTRIBUTE_MIXING_COLUMNS])
    mixing_path = None
    if mixing_attributes:
        mixing_df = pd.DataFrame.from_records([
            {
                "condition": condition,
                **attribute_mixing(
                    graph, attribute,
                    kind=_attribute_kind(
                        nodes_df[attribute], domain_profile.node_attributes.get(attribute, {}).get("kind", "")
                    ),
                    permutations=permutations, seed=random_seed,
                ),
            }
            for condition, graph in graphs.items()
            for attribute in mixing_attributes
        ])
        mixing_path = output_dir / "attribute_mixing.csv"
        mixing_df.to_csv(mixing_path, index=False)

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
    if nodes_df is not None:
        attribute_cols = [c for c in nodes_df.columns if c not in ("node", node_condition_col)]
        report_lines.extend(
            _node_table_lines(node_stats, unmatched_node_rows, attribute_cols, _describe_data_source(node_table))
        )
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
            "compare against randomized reference graphs and aren't well-defined otherwise. They're "
            f"also skipped (NaN) above {MAX_SMALL_WORLD_NODES:,} nodes, where rewiring and re-measuring "
            "the graph many times is infeasible."
        )
        report_lines.append("")
    if any(graph.number_of_nodes() > MAX_GREEDY_COMMUNITY_NODES for graph in graphs.values()):
        report_lines.append(
            f"Note: for networks over {MAX_GREEDY_COMMUNITY_NODES:,} nodes, communities (and so modularity) come from "
            "the Louvain method (Blondel et al., 2008), seeded by --random-seed, instead of greedy modularity "
            "maximization, which is infeasible at that size. Louvain usually finds higher modularity, so compare "
            "modularity only between networks analyzed with the same method."
        )
        report_lines.append("")
    if "path_length_sources" in metrics_df.columns and metrics_df["path_length_sources"].notna().any():
        report_lines.append(
            "Note: for networks too large for exact all-pairs paths, avg_shortest_path_length is estimated by "
            "breadth-first search from randomly chosen sources (path_length_sources, seeded by --random-seed). "
            "Each source's distances are exact, so the estimate's error comes only from which sources were "
            "drawn; rerun with another seed to see its size."
        )
        report_lines.append("")
    report_lines.append("## Network metrics")
    report_lines.extend(_as_code_block(metrics_df.to_string(index=False)))
    if compute_centrality and not centrality_df.empty:
        report_lines.append("## Node centrality")
        report_lines.extend(_as_code_block(centrality_df.to_string(index=False)))
    if compute_communities and not communities_df.empty:
        report_lines.append("## Communities")
        report_lines.append(
            "Detected by greedy modularity maximization; this is the partition the modularity metric "
            f"scores. Per-node assignments are in {communities_path.name}."
        )
        report_lines.append("")
        for condition, frame in communities_df.groupby("condition", sort=False):
            sizes = frame.drop_duplicates("community").sort_values("community")["community_size"].tolist()
            shown = ", ".join(str(size) for size in sizes[:10]) + (", ..." if len(sizes) > 10 else "")
            report_lines.append(f"- {condition}: {len(sizes)} communities, sizes {shown}")
        report_lines.append("")
    if mixing_path is not None:
        report_lines.extend(_attribute_mixing_lines(mixing_df, domain_profile, permutations, mixing_path))
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
        "node_communities": communities_df,
        "node_table": nodes_df if nodes_df is not None else pd.DataFrame(),
        "node_table_summary": node_stats,
        "attribute_mixing": mixing_df,
        "attribute_mixing_csv": str(mixing_path) if mixing_path else None,
        "metrics_csv": str(metrics_path),
        "centrality_csv": str(centrality_path) if centrality_path else None,
        "communities_csv": str(communities_path) if communities_path else None,
        "degree_plot_path": degree_plot_path,
        "report_md": str(report_path),
        "report_html": html_path,
    }
