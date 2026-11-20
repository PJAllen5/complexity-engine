"""Known-answer validation of the animal_social_network profile on the Doubtful Sound dolphins,
roadmap item A3 (docs/design/validation-and-new-domains.md).

Published:
- Lusseau & Newman (2004, arXiv q-bio/0403029): 62 dolphins and 159 ties, all connected;
  assortative mixing by degree r = -0.044 +/- 0.080 (a null result); SN100 has the highest
  betweenness, and its disappearance coincided with the community splitting in two.
- Newman & Girvan (2004, Phys Rev E 69:026113, Fig. 11): modularity Q = 0.38 +/- 0.08 for the
  split into two groups, peaking at 0.52 +/- 0.03 with the larger group's four subgroups.

Criteria, fixed before any statistic was computed on this file:
1. Exactly 62 nodes and 159 edges, one connected component.
2. Degree assortativity within 0.005 of -0.044 (same network, so only rounding should differ).
3. SN100 has the highest betweenness centrality.
4. Modularity within 0.52 +/- 0.03, the paper's own uncertainty. The engine's greedy
   modularity algorithm differs from Newman & Girvan's betweenness method.

Not checked: which dolphins fall in which group, and mixing by sex. The papers give those only
in figures, and reading 62 labels off a figure would make an unreliable known answer.

The data isn't committed (see data/fetch_reference_networks.py), so this file is skipped
unless the CSVs exist under data/external/.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from complexity_engine.network_analysis import analyze_network

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
EDGES = EXTERNAL / "dolphins_edges.csv"
NAMES = EXTERNAL / "dolphins_nodes.csv"

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (EDGES.exists() and NAMES.exists()),
        reason="Dolphin network not downloaded; run python data/fetch_reference_networks.py",
    ),
]


@pytest.fixture(scope="module")
def dolphins(tmp_path_factory):
    return analyze_network(
        EDGES,
        node_table=NAMES,
        condition_col=None,
        weight_col=None,
        domain="animal_social_network",
        compute_centrality=True,
        output_dir=tmp_path_factory.mktemp("dolphins"),
    )


def test_network_has_the_published_size(dolphins):
    metrics = dolphins["network_metrics_summary"].iloc[0]
    assert (metrics["n_nodes"], metrics["n_edges"], metrics["n_connected_components"]) == (62, 159, 1)
    assert dolphins["node_table_summary"].iloc[0]["isolates_added"] == 0


def test_no_assortative_mixing_by_degree(dolphins):
    assert nx.degree_assortativity_coefficient(dolphins["graphs"]["all"]) == pytest.approx(-0.044, abs=0.005)


def test_sn100_is_the_main_broker(dolphins):
    centrality = dolphins["node_centrality"].sort_values("betweenness_centrality", ascending=False)
    assert centrality.iloc[0]["label"] == "SN100"


def test_modularity_matches_newman_and_girvan(dolphins):
    assert dolphins["network_metrics_summary"].iloc[0]["modularity"] == pytest.approx(0.52, abs=0.03)
