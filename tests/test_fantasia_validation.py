"""Known-answer validation of the time-series pipeline on real heart-rate data.

Runs the full pipeline on the PhysioNet Fantasia database (20 young, 20 elderly healthy
subjects, 120 minutes of resting RR intervals each) and checks the age effects reported
in the literature on this data:

- Sample entropy is lower in the elderly (Kaplan et al., 1991, with approximate entropy;
  Balasubramanian & Nagaraj, 2016, with sample entropy on this database).
- The short-range DFA exponent is higher in the elderly (Iyengar et al., 1996:
  1.12 +/- 0.19 vs 0.90 +/- 0.14).
- The long-range DFA exponent is lower in the elderly (Iyengar et al., 1996:
  0.75 +/- 0.17 vs 0.99 +/- 0.10). This is NOT reproduced over the 16-64 beat window
  fixed in advance; see the strict xfail below.

The data isn't committed (see data/fetch_fantasia.py), so this file is skipped unless
data/external/fantasia_rr.csv exists. Run `python data/fetch_fantasia.py` first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from complexity_engine.analysis import analyze_dataset

FANTASIA_CSV = Path(__file__).resolve().parents[1] / "data" / "external" / "fantasia_rr.csv"

pytestmark = pytest.mark.skipif(
    not FANTASIA_CSV.exists(), reason="Fantasia data not downloaded; run python data/fetch_fantasia.py"
)


@pytest.fixture(scope="module")
def significance(tmp_path_factory):
    result = analyze_dataset(
        FANTASIA_CSV,
        signal_cols=["rr_interval"],
        condition_col="condition",
        domain="heart_rate_variability",
        output_dir=tmp_path_factory.mktemp("fantasia"),
    )
    rows = result["significance_summary"]
    assert set(rows["test"]) == {"welch_on_recording_values"}
    assert (rows["n_a"] == 20).all() and (rows["n_b"] == 20).all()
    return rows.set_index("measure")


def _elderly_vs_young(significance, measure):
    row = significance.loc[measure]
    assert (row["condition_a"], row["condition_b"]) == ("elderly", "young")
    return row["mean_a"], row["mean_b"], row["p_value"]


def test_sample_entropy_is_lower_in_elderly(significance):
    elderly, young, p_value = _elderly_vs_young(significance, "sample_entropy")
    assert elderly < young
    assert p_value < 0.05


def test_short_range_dfa_is_higher_in_elderly(significance):
    elderly, young, p_value = _elderly_vs_young(significance, "dfa_short_term")
    assert elderly > young
    assert p_value < 0.05


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Not reproduced over 16-64 beats (elderly 1.03 vs young 1.01, p=0.58). The published "
        "fitting window couldn't be verified, and an exploratory scan showed the result depends "
        "on the window, reversing direction between 32-128 and 100-1000 beats."
    ),
)
def test_long_range_dfa_is_lower_in_elderly(significance):
    elderly, young, p_value = _elderly_vs_young(significance, "dfa_long_term")
    assert elderly < young
    assert p_value < 0.05
