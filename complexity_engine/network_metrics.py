"""Network-science metrics for graph-shaped data (edge lists), as distinct from the
time-series pipeline in analysis.py/metrics.py. See network_analysis.py for the
ingestion and reporting layer built on top of this module.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

# Every key compute_network_metrics() can produce, for validating a domain profile's
# recommended_metrics before an analysis run (see network_analysis.py). The
# directed-only keys are included so a directed-specific profile validates too;
# they just won't appear in an undirected run's output columns.
NETWORK_METRIC_KEYS = frozenset(
    {
        "n_nodes",
        "n_edges",
        "density",
        "mean_degree",
        "avg_clustering",
        "transitivity",
        "modularity",
        "avg_shortest_path_length",
        "n_connected_components",
        "mean_in_degree",
        "mean_out_degree",
        "n_strongly_connected_components",
        "small_world_sigma",
        "small_world_omega",
    }
)


def build_graph(
    edges: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = "weight",
    directed: bool = False,
) -> nx.Graph:
    """Build a graph from an edge list.

    Undirected by default: Watts-Strogatz small-world and Barabasi-Albert
    scale-free structure -- the theory this module's metrics are grounded in --
    are both defined on undirected graphs. Pass directed=True for data with a
    genuine directionality (e.g. directed functional/causal influence).
    """
    graph = nx.DiGraph() if directed else nx.Graph()
    has_weight = weight_col is not None and weight_col in edges.columns
    for _, row in edges.iterrows():
        source, target = row[source_col], row[target_col]
        weight = float(row[weight_col]) if has_weight else 1.0
        graph.add_edge(source, target, weight=weight)
    return graph


def _empty_metrics() -> dict[str, float]:
    return {
        "n_nodes": 0, "n_edges": 0, "density": float("nan"), "mean_degree": float("nan"),
        "avg_clustering": float("nan"), "transitivity": float("nan"), "modularity": float("nan"),
        "avg_shortest_path_length": float("nan"), "n_connected_components": 0,
    }


def compute_network_metrics(graph: nx.Graph) -> dict[str, float]:
    """Compute graph-level summary metrics.

    avg_shortest_path_length is unweighted hop count (the standard definition used
    in small-world literature). avg_clustering is weighted when the graph has edge
    weights; modularity uses weights too (community detection over connection
    strength is a standard use of edge weight, unlike treating a strength as a
    literal distance).

    For a directed graph, mean_in_degree/mean_out_degree are reported alongside
    mean_degree (total in+out), n_connected_components counts *weakly* connected
    components (is the graph one piece if you ignore direction), and
    avg_shortest_path_length is computed on the largest *strongly* connected
    component (the only subgraph where every pair is mutually reachable, so
    directed path length is well-defined) -- see n_strongly_connected_components.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return _empty_metrics()

    is_directed = graph.is_directed()
    has_weight = any("weight" in data for _, _, data in graph.edges(data=True))

    if is_directed:
        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        largest_strong = graph.subgraph(max(strong_components, key=len)) if strong_components else graph
        avg_shortest_path_length = (
            nx.average_shortest_path_length(largest_strong) if largest_strong.number_of_nodes() > 1 else float("nan")
        )
        n_connected_components = len(weak_components)
    else:
        components = list(nx.connected_components(graph))
        largest_component = graph.subgraph(max(components, key=len)) if components else graph
        avg_shortest_path_length = (
            nx.average_shortest_path_length(largest_component) if largest_component.number_of_nodes() > 1 else float("nan")
        )
        n_connected_components = len(components)

    communities = detect_communities(graph)
    modularity = nx.algorithms.community.modularity(graph, communities, weight="weight" if has_weight else None)

    degrees = [d for _, d in graph.degree()]

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph),
        "mean_degree": sum(degrees) / n_nodes,
        "avg_clustering": nx.average_clustering(graph, weight="weight" if has_weight else None),
        "transitivity": nx.transitivity(graph),
        "modularity": modularity,
        "avg_shortest_path_length": avg_shortest_path_length,
        "n_connected_components": n_connected_components,
    }
    if is_directed:
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        metrics["mean_in_degree"] = sum(in_degrees) / n_nodes
        metrics["mean_out_degree"] = sum(out_degrees) / n_nodes
        metrics["n_strongly_connected_components"] = len(strong_components)
    return metrics


def detect_communities(graph: nx.Graph) -> list[frozenset]:
    """Communities found by greedy modularity maximization (Clauset, Newman & Moore), largest
    first. Weighted when the graph has edge weights, matching how modularity is scored."""
    if graph.number_of_nodes() == 0:
        return []
    has_weight = any("weight" in data for _, _, data in graph.edges(data=True))
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight" if has_weight else None)
    return sorted(communities, key=len, reverse=True)


def compute_node_communities(graph: nx.Graph) -> pd.DataFrame:
    """Each node's community (0 = largest) from detect_communities(), the same partition the
    modularity metric scores, so it can be checked against known groups in the data."""
    records = [
        {"node": node, "community": index, "community_size": len(members)}
        for index, members in enumerate(detect_communities(graph))
        for node in members
    ]
    return pd.DataFrame.from_records(records, columns=["node", "community", "community_size"])


def compute_node_centrality(graph: nx.Graph) -> pd.DataFrame:
    """Per-node degree, betweenness, and eigenvector centrality.

    Eigenvector centrality can fail to converge on some graphs (e.g. graphs with
    disconnected components); that column is left NaN for such graphs rather than
    raising, so a single hard-to-analyze graph doesn't take down the whole report.
    """
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "degree_centrality", "betweenness_centrality", "eigenvector_centrality"])

    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, weight="weight")
    except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution, nx.NetworkXError):
        eigenvector_centrality = {node: float("nan") for node in graph.nodes()}

    return pd.DataFrame.from_records(
        [
            {
                "node": node,
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "eigenvector_centrality": eigenvector_centrality[node],
            }
            for node in graph.nodes()
        ]
    )


def compute_small_world_metrics(graph: nx.Graph, seed: int = 42) -> dict[str, float]:
    """Compute the small-world sigma (Humphries & Gurney 2008) and omega (Telesford
    et al. 2011) coefficients, comparing the graph's clustering and path length
    against randomized reference graphs with the same degree sequence.

    sigma > 1 indicates small-world structure (elevated clustering without a
    proportional increase in path length, relative to a matched random graph).
    omega near 0 indicates small-world structure; negative is lattice-like;
    positive is random-like.

    Requires an undirected, *fully connected* graph with >= 4 nodes and >= 2 edges
    -- every precondition is checked up front, returning NaN rather than computing,
    because nx.sigma()/nx.omega() call nx.random_reference() internally, and that
    function can hang indefinitely (confirmed empirically, not just in theory) on a
    graph with an isolated node: its edge-rewiring search can never terminate
    normally since a degree-0 node can never gain an edge through swaps, and the
    retry loop that searches for a valid swap has no escape for that case. This is
    the only place in the pipeline that could hang without this guard.

    This is computationally expensive (generates several randomized reference
    graphs) and nondeterministic without a fixed seed, so it's opt-in in
    network_analysis.py rather than always computed.
    """
    empty = {"small_world_sigma": float("nan"), "small_world_omega": float("nan")}
    if graph.is_directed():
        return empty
    if graph.number_of_nodes() < 4 or graph.number_of_edges() < 2:
        return empty
    if not nx.is_connected(graph):
        return empty

    try:
        sigma = nx.sigma(graph, seed=seed)
        omega = nx.omega(graph, seed=seed)
    except nx.NetworkXError:
        return empty
    return {"small_world_sigma": float(sigma), "small_world_omega": float(omega)}
