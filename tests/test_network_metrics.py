import math
import subprocess
import sys

import networkx as nx
import pandas as pd

from complexity_engine.network_metrics import (
    build_graph,
    compute_network_metrics,
    compute_node_centrality,
    compute_node_communities,
    compute_small_world_metrics,
)


def test_build_graph_creates_expected_nodes_and_edges():
    edges = pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"], "weight": [1.0, 2.0, 3.0]})
    graph = build_graph(edges)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
    assert graph["a"]["b"]["weight"] == 1.0


def test_build_graph_defaults_weight_to_one_without_weight_column():
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = build_graph(edges, weight_col=None)
    assert graph["a"]["b"]["weight"] == 1.0


def test_compute_network_metrics_on_empty_graph():
    metrics = compute_network_metrics(nx.Graph())
    assert metrics["n_nodes"] == 0
    assert metrics["n_edges"] == 0


def test_compute_network_metrics_triangle_has_full_clustering():
    edges = pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"]})
    graph = build_graph(edges, weight_col=None)

    metrics = compute_network_metrics(graph)

    assert metrics["n_nodes"] == 3
    assert metrics["n_edges"] == 3
    assert metrics["avg_clustering"] == 1.0
    assert metrics["transitivity"] == 1.0
    assert metrics["n_connected_components"] == 1
    assert metrics["avg_shortest_path_length"] == 1.0


def test_compute_network_metrics_detects_disconnected_components():
    edges = pd.DataFrame({"source": ["a", "c"], "target": ["b", "d"]})
    graph = build_graph(edges, weight_col=None)

    metrics = compute_network_metrics(graph)

    assert metrics["n_connected_components"] == 2


def test_hub_dominated_graph_has_higher_max_degree_than_uniform_graph():
    uniform_graph = nx.watts_strogatz_graph(n=20, k=4, p=0.1, seed=1)
    hub_graph = nx.barabasi_albert_graph(n=20, m=2, seed=1)

    uniform_max_degree = max(d for _, d in uniform_graph.degree())
    hub_max_degree = max(d for _, d in hub_graph.degree())

    assert hub_max_degree > uniform_max_degree


def test_build_graph_directed_preserves_edge_direction():
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = build_graph(edges, weight_col=None, directed=True)

    assert graph.is_directed()
    assert graph.has_edge("a", "b")
    assert not graph.has_edge("b", "a")


def test_compute_network_metrics_directed_reports_in_out_degree():
    edges = pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"]})
    graph = build_graph(edges, weight_col=None, directed=True)

    metrics = compute_network_metrics(graph)

    assert metrics["mean_in_degree"] == 1.0
    assert metrics["mean_out_degree"] == 1.0
    assert metrics["n_strongly_connected_components"] == 1
    # a->b->c->a: 1-hop pairs (a,b),(b,c),(c,a) and 2-hop pairs (a,c),(b,a),(c,b) -> mean 1.5
    assert metrics["avg_shortest_path_length"] == 1.5


def test_compute_network_metrics_directed_without_cycle_has_nan_path_length():
    # a->b->c: not strongly connected (no way back from c to a), so the largest
    # strongly connected component is a single node and path length is undefined.
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = build_graph(edges, weight_col=None, directed=True)

    metrics = compute_network_metrics(graph)

    assert metrics["n_strongly_connected_components"] == 3
    assert pd.isna(metrics["avg_shortest_path_length"])


def test_compute_node_centrality_returns_expected_columns():
    edges = pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"]})
    graph = build_graph(edges, weight_col=None)

    centrality = compute_node_centrality(graph)

    assert set(centrality["node"]) == {"a", "b", "c"}
    assert list(centrality.columns) == ["node", "degree_centrality", "betweenness_centrality", "eigenvector_centrality"]
    assert (centrality["degree_centrality"] == 1.0).all()  # triangle: every node connects to both others


def test_compute_node_centrality_on_empty_graph_returns_empty_frame():
    centrality = compute_node_centrality(nx.Graph())
    assert centrality.empty


def test_compute_small_world_metrics_on_connected_graph():
    graph = nx.watts_strogatz_graph(n=12, k=4, p=0.1, seed=1)

    result = compute_small_world_metrics(graph, seed=1)

    assert math.isfinite(result["small_world_sigma"])
    assert math.isfinite(result["small_world_omega"])


def test_compute_small_world_metrics_is_reproducible_with_seed():
    graph = nx.watts_strogatz_graph(n=12, k=4, p=0.1, seed=1)

    first = compute_small_world_metrics(graph, seed=7)
    second = compute_small_world_metrics(graph, seed=7)

    assert first == second


def test_compute_small_world_metrics_rejects_directed_graph():
    graph = nx.watts_strogatz_graph(n=12, k=4, p=0.1, seed=1).to_directed()

    result = compute_small_world_metrics(graph)

    assert math.isnan(result["small_world_sigma"])
    assert math.isnan(result["small_world_omega"])


def test_compute_small_world_metrics_rejects_tiny_graph():
    edges = pd.DataFrame({"source": ["a"], "target": ["b"]})
    graph = build_graph(edges, weight_col=None)

    result = compute_small_world_metrics(graph)

    assert math.isnan(result["small_world_sigma"])
    assert math.isnan(result["small_world_omega"])


def test_compute_small_world_metrics_disconnected_graph_does_not_hang():
    """Regression test for a real bug found while building this feature: nx.sigma()/
    nx.omega() call nx.random_reference() internally, which can hang indefinitely on
    a graph with an isolated node (its edge-rewiring search can never terminate
    normally, since a degree-0 node can never gain an edge through swaps). Runs in a
    subprocess with a real timeout, so a regression in the precondition guard fails
    this test instead of hanging the whole suite.
    """
    code = (
        "import math\n"
        "import networkx as nx\n"
        "from complexity_engine.network_metrics import compute_small_world_metrics\n"
        "g = nx.Graph()\n"
        "g.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])\n"
        "g.add_node('isolated')\n"
        "result = compute_small_world_metrics(g)\n"
        "assert math.isnan(result['small_world_sigma'])\n"
        "assert math.isnan(result['small_world_omega'])\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def _two_triangles_joined_by_a_bridge():
    return nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x"), ("c", "x")])


def test_compute_node_communities_splits_two_triangles_at_the_bridge():
    graph = _two_triangles_joined_by_a_bridge()
    communities = compute_node_communities(graph)
    assert list(communities.columns) == ["node", "community", "community_size"]
    groups = communities.groupby("community")["node"].apply(frozenset)
    assert set(groups) == {frozenset("abc"), frozenset("xyz")}
    assert (communities["community_size"] == 3).all()


def test_node_communities_are_the_partition_modularity_scores():
    graph = _two_triangles_joined_by_a_bridge()
    communities = compute_node_communities(graph).groupby("community")["node"].apply(set).tolist()
    expected = nx.algorithms.community.modularity(graph, communities)
    assert compute_network_metrics(graph)["modularity"] == expected


def test_compute_node_communities_on_empty_graph():
    assert compute_node_communities(nx.Graph()).empty
