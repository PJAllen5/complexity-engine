"""Known-answer validation of the network pipeline on the two small-world networks of
Watts & Strogatz (1998), roadmap items A1 and A2 (docs/design/validation-and-new-domains.md).

Published values (Watts & Strogatz, 1998, Table 1). L is the average shortest path length,
C the average clustering coefficient, and the random values are for a random graph with the
same n and mean degree k:

    network        n      k      L       L_random   C       C_random
    C. elegans     282    14     2.65    2.25       0.28    0.05
    power grid     4941   2.67   18.7    12.4       0.080   0.005

Criteria, fixed before any statistic was computed on these files. In them, the random values
are the standard estimates L_random ~ ln n / ln k and C_random ~ k / n, computed for each
file's own n and k:
- Power grid: exactly 4,941 nodes and 6,594 edges (Newman's published counts for this
  version), C within 0.01 of 0.080, L within 10% of 18.7, C over 10x C_random, and L under
  2x L_random (the grid's L is the largest relative to random of the three W&S networks).
- C. elegans: Newman's version has 297 neurons, not the paper's 282, so the published values
  get wider tolerances: C within 0.05 of 0.28, L within 10% of 2.65. Small world: C over
  3x C_random and L under 1.2x L_random, both computed for this version's own n and k.
  sigma > 1 (Humphries & Gurney, 2008) is checked on C. elegans only, where it's cheap
  enough; the grid uses W&S's own random-graph estimates instead.

The data isn't committed (see data/fetch_reference_networks.py), so this file is skipped
unless the CSVs exist under data/external/.

Result: everything passes except the power grid's L < 2x L_random (strict xfail below). That
criterion was wrong: W&S's random values didn't come from ln n / ln k, which gives 8.66 for
the grid against their 12.4. See docs/validation/reference_networks.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from complexity_engine.network_analysis import analyze_network

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
CELEGANS = EXTERNAL / "celegans_edges.csv"
POWER_GRID = EXTERNAL / "power_grid_edges.csv"

pytestmark = pytest.mark.skipif(
    not (CELEGANS.exists() and POWER_GRID.exists()),
    reason="Reference networks not downloaded; run python data/fetch_reference_networks.py",
)


def _metrics(path, domain, tmp_path_factory, small_world=False):
    result = analyze_network(
        path,
        condition_col=None,
        weight_col=None,
        domain=domain,
        compute_small_world=small_world,
        output_dir=tmp_path_factory.mktemp(path.stem),
    )
    return result["network_metrics_summary"].iloc[0]


def _random_graph_estimates(metrics):
    n, k = metrics["n_nodes"], metrics["mean_degree"]
    return np.log(n) / np.log(k), k / n


@pytest.fixture(scope="module")
def power_grid(tmp_path_factory):
    return _metrics(POWER_GRID, "network", tmp_path_factory)


@pytest.fixture(scope="module")
def celegans(tmp_path_factory):
    return _metrics(CELEGANS, "neuronal_network", tmp_path_factory, small_world=True)


def test_power_grid_is_the_published_network(power_grid):
    assert (power_grid["n_nodes"], power_grid["n_edges"]) == (4941, 6594)
    assert power_grid["n_connected_components"] == 1


def test_power_grid_reproduces_watts_strogatz(power_grid):
    assert power_grid["avg_clustering"] == pytest.approx(0.080, abs=0.01)
    assert power_grid["avg_shortest_path_length"] == pytest.approx(18.7, rel=0.10)


def test_power_grid_is_far_more_clustered_than_random(power_grid):
    _, c_random = _random_graph_estimates(power_grid)
    assert power_grid["avg_clustering"] > 10 * c_random


@pytest.mark.xfail(
    strict=True,
    reason=(
        "L = 18.99 against a limit of 2 x ln(n)/ln(k) = 17.3. The criterion assumed W&S's random "
        "path length came from ln n / ln k, but that gives 8.66 at k = 2.67 versus their 12.4; "
        "near the percolation threshold the formula underestimates random-graph path length. "
        "Against the published 12.4 the ratio is 1.53, but that comparison was chosen after the run."
    ),
)
def test_power_grid_path_length_is_close_to_random(power_grid):
    l_random, _ = _random_graph_estimates(power_grid)
    assert power_grid["avg_shortest_path_length"] < 2 * l_random


def test_celegans_is_newmans_297_neuron_version(celegans):
    assert celegans["n_nodes"] == 297


def test_celegans_reproduces_watts_strogatz(celegans):
    assert celegans["avg_clustering"] == pytest.approx(0.28, abs=0.05)
    assert celegans["avg_shortest_path_length"] == pytest.approx(2.65, rel=0.10)


def test_celegans_is_a_small_world(celegans):
    l_random, c_random = _random_graph_estimates(celegans)
    assert celegans["avg_clustering"] > 3 * c_random
    assert celegans["avg_shortest_path_length"] < 1.2 * l_random
    assert celegans["small_world_sigma"] > 1
