"""Known-answer validation of the time-series pipeline on real climate data.

Runs the full pipeline, with the climate_temperature domain profile, on deseasonalized daily
maximum temperatures from GHCN-Daily stations named by Eichner et al. (2003), and checks
their published findings:

- Continental (inland and coastal) stations show long-range persistence with a DFA exponent
  close to 0.65 and a very narrow spread (Koscielny-Bunde et al., 1998; Eichner et al., 2003).
- Island stations show a higher exponent, averaging about 0.8 and ranging 0.65-0.85.
  The direction is reproduced, but with 5 qualifying island stations the difference is not
  significant at p < 0.05; see the strict xfail below.

Settings (DFA2 over 30-3000 days) and inclusion rules were fixed before any exponent was
computed. The data isn't committed (see data/fetch_ghcn_temperature.py), so this file is
skipped unless data/external/ghcn_tmax_anomaly.csv exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from complexity_engine.analysis import analyze_dataset

GHCN_CSV = Path(__file__).resolve().parents[1] / "data" / "external" / "ghcn_tmax_anomaly.csv"

pytestmark = pytest.mark.skipif(
    not GHCN_CSV.exists(), reason="GHCN data not downloaded; run python data/fetch_ghcn_temperature.py"
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return analyze_dataset(
        GHCN_CSV,
        signal_cols=["tmax_anomaly"],
        condition_col="condition",
        unit_cols=["station_id"],
        domain="climate_temperature",
        output_dir=tmp_path_factory.mktemp("ghcn"),
    )


@pytest.fixture(scope="module")
def dfa_by_station(result):
    units = result["complexity_by_unit"]
    assert units["dfa_exponent"].notna().all()
    return {condition: group["dfa_exponent"] for condition, group in units.groupby("condition")}


@pytest.fixture(scope="module")
def dfa_test(result):
    rows = result["significance_summary"].set_index("measure")
    row = rows.loc["dfa_exponent"]
    assert row["test"] == "welch_on_recording_values"
    assert (row["condition_a"], row["condition_b"], row["n_a"], row["n_b"]) == ("continental", "island", 13, 5)
    return row


def test_continental_stations_share_an_exponent_near_065(dfa_by_station):
    continental = dfa_by_station["continental"]
    # Published: ~0.65. Measured 0.684 +/- 0.032; the 30-day lower window sits close enough
    # to the DFA2 crossover to lift values slightly (see docs/validation/ghcn_temperature.md).
    assert continental.mean() == pytest.approx(0.65, abs=0.05)
    assert continental.std() < 0.05
    assert continental.min() > 0.55  # long-range persistent everywhere, not white noise (0.5)


def test_island_stations_are_more_persistent_and_more_varied(dfa_by_station, dfa_test):
    island, continental = dfa_by_station["island"], dfa_by_station["continental"]
    assert dfa_test["mean_b"] > dfa_test["mean_a"]
    assert island.mean() > continental.mean() + 0.04
    assert island.std() > continental.std()
    assert island.min() > 0.6 and island.max() < 0.9  # published range 0.65-0.85


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Direction and effect size reproduced (island 0.748 vs continental 0.684, d = 1.6), but "
        "with 5 island stations the pre-specified two-sided Welch test gives p = 0.075. An "
        "exploratory scan keeps the direction in every setting tried, with p from 0.05 to 0.66."
    ),
)
def test_island_difference_is_significant(dfa_test):
    assert dfa_test["p_value"] < 0.05


# --- All 23 stations with gap_policy: segment (#59, PR 2) ---------------------------------
# Fixed before the run: the 5 stations excluded above for coverage are added back and gaps
# handled in the engine (segment, splitting at every gap) instead. Predictions: islands
# higher at p < 0.05; continental mean 0.65 +/- 0.05 with SD < 0.05; and Chita, the most
# continental station, at alpha >= 0.6 (Eichner et al., 2003) rather than <= 0.55
# (Fraedrich & Blender, 2003). Splitting at every missing day left 7 of 23 stations without
# enough long windows; see docs/validation/gap_handling.md.

GHCN_ALL_CSV = GHCN_CSV.with_name("ghcn_tmax_anomaly_all.csv")
CHITA = "RSM00030758"


@pytest.fixture(scope="module")
def all_stations(tmp_path_factory):
    if not GHCN_ALL_CSV.exists():
        pytest.skip("Run python data/fetch_ghcn_temperature.py --all-stations")
    import yaml

    from complexity_engine.domain_profiles import get_domain_profile

    profile = {**get_domain_profile("climate_temperature").as_dict(), "name": "climate_segment", "gap_policy": "segment"}
    config = tmp_path_factory.mktemp("config") / "segment.yaml"
    config.write_text(yaml.safe_dump({"climate_segment": profile}), encoding="utf-8")
    return analyze_dataset(
        GHCN_ALL_CSV,
        signal_cols=["tmax_anomaly"],
        unit_cols=["station_id"],
        domain="climate_segment",
        domain_config=config,
        output_dir=tmp_path_factory.mktemp("ghcn_all"),
    )


def test_all_stations_continental_exponent_near_065(all_stations):
    units = all_stations["complexity_by_unit"]
    continental = units.loc[units["condition"] == "continental", "dfa_exponent"].dropna()
    assert continental.mean() == pytest.approx(0.65, abs=0.05)
    assert continental.std() < 0.05


def test_most_continental_station_is_persistent_not_white_noise(all_stations):
    chita = all_stations["complexity_by_unit"].set_index("station_id").loc[CHITA, "dfa_exponent"]
    assert chita >= 0.6


@pytest.mark.xfail(
    strict=True,
    reason="Islands 0.719 vs continental 0.679 (d = 1.2) but p = 0.11, with only 12 vs 4 stations defined.",
)
def test_all_stations_island_difference_is_significant(all_stations):
    row = all_stations["significance_summary"].set_index("measure").loc["dfa_exponent"]
    assert row["mean_b"] > row["mean_a"] and row["p_value"] < 0.05
