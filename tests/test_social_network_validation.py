"""Known-answer validation of the network pipeline on real social networks.

Runs the full pipeline, with the social_network domain profile, on two networks with
published results:

- Zachary's karate club (Zachary, 1977). Newman & Girvan (2004) found modularity of about
  0.4, and communities matching the two factions the club split into with at most one
  member misclassified. Modularity is reproduced. Community recovery misassigns two
  members, one more than the threshold fixed in advance; see the strict xfail below.
- ca-GrQc, arXiv GR-QC co-authorships (Leskovec et al., 2007). The size, clustering, and
  giant component match SNAP's published statistics exactly, and the network is a small
  world in Newman's (2001) sense: clustering far above, and path length close to, a random
  network of the same size and mean degree.

The data isn't committed (see data/fetch_social_networks.py), so this file is skipped
unless the CSVs exist under data/external/.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from complexity_engine.network_analysis import analyze_network

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
KARATE_EDGES = EXTERNAL / "karate_club_edges.csv"
KARATE_FACTIONS = EXTERNAL / "karate_club_factions.csv"
GRQC_EDGES = EXTERNAL / "ca_grqc_edges.csv"

pytestmark = pytest.mark.skipif(
    not (KARATE_EDGES.exists() and KARATE_FACTIONS.exists() and GRQC_EDGES.exists()),
    reason="Social network data not written; run python data/fetch_social_networks.py",
)


def _run(path, tmp_path_factory, name):
    return analyze_network(
        path,
        condition_col=None,
        weight_col=None,
        domain="social_network",
        compute_communities=True,
        output_dir=tmp_path_factory.mktemp(name),
    )


@pytest.fixture(scope="module")
def karate(tmp_path_factory):
    return _run(KARATE_EDGES, tmp_path_factory, "karate")


@pytest.fixture(scope="module")
def grqc(tmp_path_factory):
    return _run(GRQC_EDGES, tmp_path_factory, "grqc")


def _misassigned_members(karate):
    """Members whose faction differs from the majority faction of their detected community."""
    members = karate["node_communities"].merge(pd.read_csv(KARATE_FACTIONS), on="node")
    majority = members.groupby("community")["faction"].agg(lambda factions: factions.mode()[0])
    return sorted(members.loc[members["faction"] != members["community"].map(majority), "node"])


def test_karate_club_modularity_is_near_04(karate):
    metrics = karate["network_metrics_summary"].iloc[0]
    assert (metrics["n_nodes"], metrics["n_edges"]) == (34, 78)
    assert 0.35 <= metrics["modularity"] <= 0.45


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Greedy modularity finds 3 communities that nest inside the two factions except for "
        "members 9 and 10, so 2 are misassigned against a threshold of 1. Member 9 is the one "
        "Zachary's own model misassigned, and member 10 has one tie to each faction."
    ),
)
def test_karate_club_communities_match_factions(karate):
    assert len(_misassigned_members(karate)) <= 1


def test_ca_grqc_matches_snap_statistics(grqc):
    metrics = grqc["network_metrics_summary"].iloc[0]
    assert (metrics["n_nodes"], metrics["n_edges"]) == (5242, 14496)
    assert round(metrics["avg_clustering"], 4) == 0.5296
    graph = grqc["graphs"]["all"]
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    assert (giant.number_of_nodes(), giant.number_of_edges()) == (4158, 13428)
    assert sum(nx.triangles(graph).values()) // 3 == 48260


def test_ca_grqc_is_a_small_world(grqc):
    metrics = grqc["network_metrics_summary"].iloc[0]
    graph = grqc["graphs"]["all"]
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    n, mean_degree = giant.number_of_nodes(), 2 * giant.number_of_edges() / giant.number_of_nodes()
    # Random-graph estimates used by Newman (2001): clustering ~ k/n, path length ~ ln n / ln k.
    assert metrics["avg_clustering"] > 100 * mean_degree / n
    assert metrics["avg_shortest_path_length"] < 1.5 * np.log(n) / np.log(mean_degree)
