"""Large-network mode (roadmap E6): sampled path length, Louvain above a size threshold, small-world
skipped above a size threshold, and a bulk graph builder. Thresholds are lowered here so small
graphs exercise each switch."""

import networkx as nx
import pandas as pd
import pytest

from complexity_engine import network_metrics
from complexity_engine.network_analysis import analyze_network
from complexity_engine.network_metrics import (
    build_graph,
    compute_network_metrics,
    compute_small_world_metrics,
    sampled_average_path_length,
)


@pytest.fixture
def graph():
    return nx.connected_watts_strogatz_graph(400, 6, 0.1, seed=1)


def test_sampling_every_node_gives_the_exact_path_length(graph):
    assert sampled_average_path_length(graph, 400) == pytest.approx(nx.average_shortest_path_length(graph))


def test_sampled_path_length_is_close_and_seeded(graph):
    exact = nx.average_shortest_path_length(graph)
    estimate = sampled_average_path_length(graph, 40, seed=1)
    assert estimate == pytest.approx(exact, rel=0.05)
    assert estimate == sampled_average_path_length(graph, 40, seed=1)


def test_directed_sampling_matches_exact_on_a_strongly_connected_graph():
    g = nx.DiGraph(nx.cycle_graph(30).to_directed())
    assert sampled_average_path_length(g, 30) == pytest.approx(nx.average_shortest_path_length(g))


def test_path_length_is_exact_below_the_threshold_and_sampled_above(graph):
    exact = compute_network_metrics(graph)
    assert "path_length_sources" not in exact
    assert exact["avg_shortest_path_length"] == pytest.approx(nx.average_shortest_path_length(graph))
    sampled = compute_network_metrics(graph, max_exact_path_nodes=100, path_sources=25)
    assert sampled["path_length_sources"] == 25
    assert sampled["avg_shortest_path_length"] == pytest.approx(exact["avg_shortest_path_length"], rel=0.1)


def test_louvain_replaces_greedy_above_the_threshold(monkeypatch, graph):
    assert network_metrics.community_method(graph) == "greedy"
    monkeypatch.setattr(network_metrics, "MAX_GREEDY_COMMUNITY_NODES", 100)
    assert network_metrics.community_method(graph) == "louvain"
    first = network_metrics.detect_communities(graph, seed=3)
    assert first == network_metrics.detect_communities(graph, seed=3)
    assert sum(len(c) for c in first) == 400


def test_small_world_is_skipped_above_the_size_limit(graph):
    assert compute_small_world_metrics(graph, max_nodes=100) == {
        "small_world_sigma": pytest.approx(float("nan"), nan_ok=True),
        "small_world_omega": pytest.approx(float("nan"), nan_ok=True),
    }


def test_bulk_builder_keeps_ids_last_weights_and_direction():
    edges = pd.DataFrame({"source": [1, 1, 2], "target": [2, 2, 3], "weight": [0.5, 2.0, 1.5]})
    g = build_graph(edges)
    assert set(g.nodes) == {1, 2, 3} and all(isinstance(n, int) for n in g.nodes)
    assert g[1][2]["weight"] == 2.0 and g.number_of_edges() == 2
    d = build_graph(edges, directed=True)
    assert d.has_edge(1, 2) and not d.has_edge(2, 1)
    assert all(data["weight"] == 1.0 for *_, data in build_graph(edges, weight_col=None).edges(data=True))


def test_report_explains_sampling_and_louvain(monkeypatch, tmp_path, graph):
    monkeypatch.setattr(network_metrics, "MAX_EXACT_PATH_NODES", 100)
    monkeypatch.setattr("complexity_engine.network_analysis.MAX_GREEDY_COMMUNITY_NODES", 100)
    monkeypatch.setattr(network_metrics, "MAX_GREEDY_COMMUNITY_NODES", 100)
    edges = tmp_path / "edges.csv"
    pd.DataFrame(list(graph.edges()), columns=["source", "target"]).to_csv(edges, index=False)
    result = analyze_network(edges, condition_col=None, weight_col=None, path_sources=30, output_dir=tmp_path / "out")
    assert result["network_metrics_summary"].iloc[0]["path_length_sources"] == 30
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Louvain method" in report and "estimated by breadth-first search" in report
