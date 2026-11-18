"""Network-science metrics for graph-shaped data (edge lists), as distinct from the
time-series pipeline in analysis.py/metrics.py. See network_analysis.py for the
ingestion and reporting layer built on top of this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
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
    # Added in bulk, in row order, so a repeated pair keeps its last weight, as edge-by-edge
    # insertion would; .tolist() keeps each column's own type (ints stay ints even when the
    # weight column is float, which row-by-row iteration used to upcast).
    weights = edges[weight_col].astype(float).tolist() if has_weight else [1.0] * len(edges)
    graph.add_weighted_edges_from(zip(edges[source_col].tolist(), edges[target_col].tolist(), weights))
    return graph


def _empty_metrics() -> dict[str, float]:
    return {
        "n_nodes": 0, "n_edges": 0, "density": float("nan"), "mean_degree": float("nan"),
        "avg_clustering": float("nan"), "transitivity": float("nan"), "modularity": float("nan"),
        "avg_shortest_path_length": float("nan"), "n_connected_components": 0,
    }


# Above these sizes the network is "large": path length is estimated from sampled sources,
# and the small-world statistics (which rewire and re-measure the whole graph many times) are
# skipped. Both are far above the validated networks (ca-GrQc, 5,242 nodes), so their
# results are unchanged.
MAX_EXACT_PATH_NODES = 50_000
DEFAULT_PATH_SOURCES = 50
MAX_SMALL_WORLD_NODES = 5_000
# Greedy modularity (Clauset, Newman & Moore) is infeasible on millions of nodes; Louvain
# (Blondel et al., 2008) takes about 2 minutes on the 2-million-node California road network.
MAX_GREEDY_COMMUNITY_NODES = 100_000


def sampled_average_path_length(graph: nx.Graph, n_sources: int = DEFAULT_PATH_SOURCES, seed: int = 0) -> float:
    """Average shortest path length of a connected graph (strongly connected if directed),
    estimated by breadth-first search from n_sources nodes drawn at random (seeded). Each
    source's mean distance to every other node is exact, so the only error is from which
    sources were drawn; with all nodes as sources it equals the exact value."""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(nodes), size=min(n_sources, len(nodes)), replace=False)
    means = []
    for index in chosen:
        lengths = nx.single_source_shortest_path_length(graph, nodes[index])
        means.append(sum(lengths.values()) / (len(lengths) - 1))
    return float(np.mean(means))


def compute_network_metrics(
    graph: nx.Graph,
    max_exact_path_nodes: int | None = None,
    path_sources: int = DEFAULT_PATH_SOURCES,
    seed: int = 0,
) -> dict[str, float]:
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

    When that component has more than max_exact_path_nodes nodes, exact all-pairs paths are
    infeasible, so the path length is estimated from path_sources sampled sources
    (sampled_average_path_length) and path_length_sources reports how many were used.
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
        path_component = largest_strong
        n_connected_components = len(weak_components)
    else:
        components = list(nx.connected_components(graph))
        largest_component = graph.subgraph(max(components, key=len)) if components else graph
        path_component = largest_component
        n_connected_components = len(components)

    if max_exact_path_nodes is None:
        max_exact_path_nodes = MAX_EXACT_PATH_NODES
    sampled_paths = path_component.number_of_nodes() > max_exact_path_nodes
    if path_component.number_of_nodes() < 2:
        avg_shortest_path_length = float("nan")
    elif sampled_paths:
        avg_shortest_path_length = sampled_average_path_length(path_component, path_sources, seed)
    else:
        avg_shortest_path_length = nx.average_shortest_path_length(path_component)

    communities = detect_communities(graph, seed=seed)
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
    if sampled_paths:
        metrics["path_length_sources"] = min(path_sources, path_component.number_of_nodes())
    return metrics


def community_method(graph: nx.Graph) -> str:
    return "louvain" if graph.number_of_nodes() > MAX_GREEDY_COMMUNITY_NODES else "greedy"


def detect_communities(graph: nx.Graph, seed: int = 0) -> list[frozenset]:
    """Communities, largest first: greedy modularity maximization (Clauset, Newman & Moore),
    or, above MAX_GREEDY_COMMUNITY_NODES nodes, the Louvain method (Blondel et al., 2008),
    seeded. Weighted when the graph has edge weights, matching how modularity is scored."""
    if graph.number_of_nodes() == 0:
        return []
    has_weight = any("weight" in data for _, _, data in graph.edges(data=True))
    weight = "weight" if has_weight else None
    if community_method(graph) == "louvain":
        communities = nx.algorithms.community.louvain_communities(graph, weight=weight, seed=seed)
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight=weight)
    return sorted(communities, key=len, reverse=True)


def compute_node_communities(graph: nx.Graph, seed: int = 0) -> pd.DataFrame:
    """Each node's community (0 = largest) from detect_communities(), the same partition the
    modularity metric scores, so it can be checked against known groups in the data."""
    records = [
        {"node": node, "community": index, "community_size": len(members)}
        for index, members in enumerate(detect_communities(graph, seed=seed))
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


def compute_small_world_metrics(graph: nx.Graph, seed: int = 42, max_nodes: int | None = None) -> dict[str, float]:
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
    if graph.number_of_nodes() > (MAX_SMALL_WORLD_NODES if max_nodes is None else max_nodes):
        return empty  # rewiring and re-measuring a graph this size many times is infeasible
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


# Columns of the attribute-mixing table, one row per attribute (and condition).
ATTRIBUTE_MIXING_COLUMNS = (
    "attribute", "kind", "n_nodes", "n_groups", "assortativity", "same_group_tie_fraction",
    "expected_same_group_fraction", "p_value", "community_nmi",
)


def _mixing_statistics(codes: np.ndarray, sources: np.ndarray, targets: np.ndarray, n_groups: int) -> tuple[float, float, float]:
    """(assortativity, same-group fraction, expected same-group fraction) for categorical codes.

    Newman's (2003) coefficient r = (sum_i e_ii - sum_i a_i b_i) / (1 - sum_i a_i b_i), where e is
    the share of tie ends joining group i to group j, a the share of ties leaving each group and b
    the share arriving. sum_i e_ii is the same-group fraction and sum_i a_i b_i its expectation
    under random mixing with the same group sizes and degrees.
    """
    source_codes, target_codes = codes[sources], codes[targets]
    same = float(np.mean(source_codes == target_codes))
    a = np.bincount(source_codes, minlength=n_groups) / source_codes.size
    b = np.bincount(target_codes, minlength=n_groups) / target_codes.size
    expected = float(np.sum(a * b))
    r = (same - expected) / (1.0 - expected) if expected < 1.0 else float("nan")
    return r, same, expected


def _numeric_assortativity(values: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> float:
    x, y = values[sources], values[targets]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def normalized_mutual_information(labels_a: Sequence, labels_b: Sequence) -> float:
    """NMI between two labelings of the same items, normalized by the mean of their entropies
    (Danon et al., 2005): 1 for identical partitions, near 0 for unrelated ones."""
    _, a = np.unique(np.asarray(labels_a, dtype=object).astype(str), return_inverse=True)
    _, b = np.unique(np.asarray(labels_b, dtype=object).astype(str), return_inverse=True)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    nonzero = joint > 0
    mutual = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / np.outer(pa, pb)[nonzero])))
    entropy = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))  # noqa: E731
    mean_entropy = (entropy(pa) + entropy(pb)) / 2
    return mutual / mean_entropy if mean_entropy > 0 else float("nan")


def attribute_mixing(
    graph: nx.Graph,
    attribute: str,
    kind: str = "categorical",
    permutations: int = 1000,
    seed: int = 0,
) -> dict[str, float | str | int]:
    """How ties mix across a node attribute: assortativity, same-group tie share, a permutation
    p-value, and agreement between the attribute and the detected communities.

    Only nodes with a value for the attribute, and ties between two such nodes, count;
    self-loops are left out, since they don't join two nodes. A directed graph uses each tie
    once, from source to target; an undirected graph uses each tie in both directions.

    - assortativity: Newman's (2003) coefficient, categorical or numeric (Pearson over ties).
    - same_group_tie_fraction / expected_same_group_fraction (categorical only): the share of
      ties inside a group, and the share expected if ties ignored the attribute but kept group
      sizes and degrees.
    - p_value: two-sided permutation test of assortativity, shuffling the attribute across
      nodes `permutations` times (seeded): (1 + k) / (1 + permutations).
    - community_nmi (categorical only): NMI between the attribute and the communities that the
      modularity metric scores, over attributed nodes.
    """
    nodes = [n for n, value in graph.nodes(data=attribute) if value is not None and not _is_missing(value)]
    result: dict[str, float | str | int] = {
        "attribute": attribute, "kind": kind, "n_nodes": len(nodes), "n_groups": float("nan"),
        "assortativity": float("nan"), "same_group_tie_fraction": float("nan"),
        "expected_same_group_fraction": float("nan"), "p_value": float("nan"), "community_nmi": float("nan"),
    }
    index = {node: i for i, node in enumerate(nodes)}
    pairs = [(index[u], index[v]) for u, v in graph.edges() if u != v and u in index and v in index]
    if not pairs:
        return result
    sources, targets = np.array(pairs).T
    if not graph.is_directed():
        sources, targets = np.concatenate([sources, targets]), np.concatenate([targets, sources])
    raw = [graph.nodes[n][attribute] for n in nodes]
    rng = np.random.default_rng(seed)

    if kind == "numeric":
        values = np.asarray(raw, dtype=float)
        observed = _numeric_assortativity(values, sources, targets)
        null = [_numeric_assortativity(rng.permutation(values), sources, targets) for _ in range(permutations)]
        result["assortativity"] = observed
    else:
        groups, codes = np.unique(np.asarray(raw, dtype=object).astype(str), return_inverse=True)
        observed, same, expected = _mixing_statistics(codes, sources, targets, len(groups))
        null = [_mixing_statistics(rng.permutation(codes), sources, targets, len(groups))[0] for _ in range(permutations)]
        community_of = {node: i for i, members in enumerate(detect_communities(graph, seed=seed)) for node in members}
        result.update({
            "n_groups": len(groups), "assortativity": observed, "same_group_tie_fraction": same,
            "expected_same_group_fraction": expected,
            "community_nmi": normalized_mutual_information(codes, [community_of[n] for n in nodes]),
        })
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if np.isfinite(observed) and null.size:
        result["p_value"] = float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + null.size))
    return result


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
