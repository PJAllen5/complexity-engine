"""Network-science metrics for graph-shaped data (edge lists), as distinct from the
time-series pipeline in analysis.py/metrics.py. See network_analysis.py for the
ingestion and reporting layer built on top of this module.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame, source_col: str = "source", target_col: str = "target", weight_col: str | None = "weight") -> nx.Graph:
    """Build an undirected graph from an edge list.

    Watts-Strogatz small-world and Barabasi-Albert scale-free structure -- the
    theory this module's metrics are grounded in -- are both defined on undirected
    graphs, so directed edges aren't supported in this first version.
    """
    graph = nx.Graph()
    has_weight = weight_col is not None and weight_col in edges.columns
    for _, row in edges.iterrows():
        source, target = row[source_col], row[target_col]
        weight = float(row[weight_col]) if has_weight else 1.0
        graph.add_edge(source, target, weight=weight)
    return graph


def compute_network_metrics(graph: nx.Graph) -> dict[str, float]:
    """Compute graph-level summary metrics.

    avg_shortest_path_length is unweighted hop count (the standard definition used
    in small-world literature), computed on the largest connected component when
    the graph is disconnected -- see n_connected_components for whether that
    matters. avg_clustering is weighted when the graph has edge weights; modularity
    uses weights too (community detection over connection strength is a standard
    use of edge weight, unlike treating a strength as a literal distance).
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "density": float("nan"), "mean_degree": float("nan"),
            "avg_clustering": float("nan"), "transitivity": float("nan"), "modularity": float("nan"),
            "avg_shortest_path_length": float("nan"), "n_connected_components": 0,
        }

    has_weight = any("weight" in data for _, _, data in graph.edges(data=True))
    components = list(nx.connected_components(graph))
    largest_component = graph.subgraph(max(components, key=len)) if components else graph

    if largest_component.number_of_nodes() > 1:
        avg_shortest_path_length = nx.average_shortest_path_length(largest_component)
    else:
        avg_shortest_path_length = float("nan")

    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight" if has_weight else None)
    modularity = nx.algorithms.community.modularity(graph, communities, weight="weight" if has_weight else None)

    degrees = [d for _, d in graph.degree()]

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph),
        "mean_degree": sum(degrees) / n_nodes,
        "avg_clustering": nx.average_clustering(graph, weight="weight" if has_weight else None),
        "transitivity": nx.transitivity(graph),
        "modularity": modularity,
        "avg_shortest_path_length": avg_shortest_path_length,
        "n_connected_components": len(components),
    }
