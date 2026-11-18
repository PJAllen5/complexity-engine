"""Known-answer validation of attribute mixing on the political blogs network (design PR 5 in
docs/design/time-axis-and-node-attributes.md).

Published (Adamic & Glance, 2005, section 2): from the front pages of 1,494 political blogs
(759 liberal, 735 conservative) in February 2005, "91% of the links originating within either
the conservative or liberal communities stay within that community." The public polblogs
network (Newman's version, 1,490 blogs) is that link network, directed from the linking blog.

Criteria, fixed before any statistic was computed on this file:
- Directed same-side tie fraction within 0.91 +/- 0.02 (self-links excluded, since they don't
  join two blogs; repeated links between the same pair count once).
- Assortativity by leaning is positive with permutation p < 0.05.

The data isn't committed (see data/fetch_reference_networks.py), so this file is skipped
unless the CSVs exist under data/external/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from complexity_engine.network_analysis import analyze_network

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
EDGES = EXTERNAL / "polblogs_edges.csv"
NODES = EXTERNAL / "polblogs_nodes.csv"

pytestmark = pytest.mark.skipif(
    not (EDGES.exists() and NODES.exists()),
    reason="Political blogs not downloaded; run python data/fetch_reference_networks.py",
)


@pytest.fixture(scope="module")
def polblogs(tmp_path_factory):
    result = analyze_network(
        EDGES,
        node_table=NODES,
        attributes=["leaning"],
        directed=True,
        condition_col=None,
        weight_col=None,
        domain="social_network",
        output_dir=tmp_path_factory.mktemp("polblogs"),
    )
    return result, result["attribute_mixing"].iloc[0]


def test_every_blog_has_a_leaning(polblogs):
    result, mixing = polblogs
    assert mixing["kind"] == "categorical" and mixing["n_groups"] == 2
    assert mixing["n_nodes"] == result["network_metrics_summary"].iloc[0]["n_nodes"] == 1490


def test_91_percent_of_links_stay_on_their_side(polblogs):
    _, mixing = polblogs
    assert mixing["same_group_tie_fraction"] == pytest.approx(0.91, abs=0.02)


def test_mixing_by_leaning_is_assortative_beyond_chance(polblogs):
    _, mixing = polblogs
    assert mixing["assortativity"] > 0 and mixing["p_value"] < 0.05
