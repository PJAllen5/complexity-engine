"""Scale test of the network pipeline's large-network mode on California's road network, roadmap
items E6 and B5 (docs/design/validation-and-new-domains.md).

Published (SNAP, roadNet-CA; Leskovec et al., 2009): 1,965,206 nodes, 2,766,607 edges, largest
connected component 1,957,027 nodes, average clustering 0.0464, 120,676 triangles, 90-percentile
effective diameter about 500. No published mean shortest path or modularity.

Criteria, fixed before the pipeline was run on this file (networkx's own clustering and triangle
count were seen while checking feasibility):
1. Exact node, edge and largest-component counts; clustering 0.0464 to 4 decimals; 120,676 triangles.
2. The sampled path length (20 sources) is stable: a second seed agrees within 5%.
3. The mean path is below the 90-percentile effective diameter (500).
4. The whole pipeline run finishes within 10 minutes.

Result: all pass except 2 (strict xfail): two seeds with 20 sources differ by 6.4%. See
docs/validation/road_network.md.

The data isn't committed (see data/fetch_road_network.py), so this file is skipped unless the
CSV exists under data/external/.
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import pytest

from complexity_engine.network_analysis import analyze_network
from complexity_engine.network_metrics import sampled_average_path_length

EDGES = Path(__file__).resolve().parents[1] / "data" / "external" / "roadnet_ca_edges.csv"

pytestmark = pytest.mark.skipif(not EDGES.exists(), reason="Road network not downloaded; run python data/fetch_road_network.py")


@pytest.fixture(scope="module")
def roads(tmp_path_factory):
    start = time.perf_counter()
    result = analyze_network(EDGES, condition_col=None, weight_col=None, domain="infrastructure_network",
                             path_sources=20, output_dir=tmp_path_factory.mktemp("roads"))
    return result, time.perf_counter() - start


def test_counts_and_clustering_match_snap(roads):
    result, _ = roads
    metrics = result["network_metrics_summary"].iloc[0]
    graph = result["graphs"]["all"]
    assert (metrics["n_nodes"], metrics["n_edges"]) == (1965206, 2766607)
    assert len(max(nx.connected_components(graph), key=len)) == 1957027
    assert round(metrics["avg_clustering"], 4) == 0.0464
    assert sum(nx.triangles(graph).values()) // 3 == 120676


def test_sampled_path_length_is_plausible(roads):
    result, _ = roads
    metrics = result["network_metrics_summary"].iloc[0]
    assert metrics["path_length_sources"] == 20
    assert metrics["avg_shortest_path_length"] < 500


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Seeds 42 and 7 give 317.9 and 297.5 links (6.4% apart; a third 20-source sample gave 320.2). "
        "In a spatial network a source's mean distance depends strongly on where it sits, so 20 "
        "sources aren't enough for +/-5%; the default of 50 should roughly halve the spread."
    ),
)
def test_sampled_path_length_is_stable_across_seeds(roads):
    result, _ = roads
    metrics = result["network_metrics_summary"].iloc[0]
    graph = result["graphs"]["all"]
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    other_seed = sampled_average_path_length(giant, 20, seed=7)
    assert other_seed == pytest.approx(metrics["avg_shortest_path_length"], rel=0.05)


def test_runs_within_budget(roads):
    result, seconds = roads
    assert seconds < 600
    report = Path(result["report_md"]).read_text(encoding="utf-8")
    assert "Louvain method" in report and "path_length_sources" in report
