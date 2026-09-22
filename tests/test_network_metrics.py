import pandas as pd

from complexity_engine.network_metrics import build_graph, compute_network_metrics


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
    import networkx as nx

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
    import networkx as nx

    uniform_graph = nx.watts_strogatz_graph(n=20, k=4, p=0.1, seed=1)
    hub_graph = nx.barabasi_albert_graph(n=20, m=2, seed=1)

    uniform_max_degree = max(d for _, d in uniform_graph.degree())
    hub_max_degree = max(d for _, d in hub_graph.degree())

    assert hub_max_degree > uniform_max_degree
