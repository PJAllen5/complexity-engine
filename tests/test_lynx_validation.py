"""Known-answer validation of the population_dynamics profile on the Canadian lynx series,
roadmap item B3 (docs/design/validation-and-new-domains.md).

Published: Hudson's Bay Company lynx returns show a cycle of about ten years (Elton &
Nicholson, 1942); Campbell & Walker (1977) surveyed the many analyses of this 1821-1934 series.

Criteria, fixed before any metric was computed on the series:
1. 114 annual values, 1821-1934.
2. dominant_period between 9 and 11 years (the nearest periodogram periods are 10.4 and 9.5).
3. Against the profile's 200 shuffled surrogates, the dominant power share is higher and
   spectral entropy lower, both with p < 0.05.

The data isn't committed (see data/fetch_population_series.py), so this file is skipped
unless data/external/lynx.csv exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset

LYNX = Path(__file__).resolve().parents[1] / "data" / "external" / "lynx.csv"

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(not LYNX.exists(), reason="Lynx data not downloaded; run python data/fetch_population_series.py"),
]


@pytest.fixture(scope="module")
def lynx(tmp_path_factory):
    result = analyze_dataset(LYNX, signal_cols=["count"], unit_cols=["series_id"], domain="population_dynamics",
                             output_dir=tmp_path_factory.mktemp("lynx"))
    return result["complexity_by_unit"].iloc[0]


def test_series_is_the_published_one():
    data = pd.read_csv(LYNX)
    assert (len(data), data["time"].min(), data["time"].max()) == (114, 1821, 1934)


def test_lynx_cycle_is_about_ten_years(lynx):
    assert 9 <= lynx["dominant_period"] <= 11


def test_cycle_is_far_more_concentrated_than_in_shuffled_counts(lynx):
    assert lynx["dominant_power_share"] > lynx["dominant_power_share_surrogate_mean"]
    assert lynx["dominant_power_share_surrogate_p"] < 0.05
    assert lynx["spectral_entropy"] < lynx["spectral_entropy_surrogate_mean"]
    assert lynx["spectral_entropy_surrogate_p"] < 0.05
