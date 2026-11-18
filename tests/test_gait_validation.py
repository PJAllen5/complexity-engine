"""Known-answer validation of the gait profile on PhysioNet stride intervals, roadmap item B4
(docs/design/validation-and-new-domains.md).

Published (Hausdorff et al., 1997, J Appl Physiol 82:262-269, abstract): the stride-interval
DFA exponent is lower in healthy elderly than young adults (0.68 +/- 0.14 vs 0.87 +/- 0.15,
P < 0.003), lower in Huntington's disease than controls (0.60 +/- 0.24 vs 0.88 +/- 0.17,
P < 0.005), and in Huntington's disease it rises with functional capacity (r = 0.78).

Criteria, fixed before any exponent was computed (gait profile: DFA1 from 16 strides to a
quarter of the series; strides over 3 SD from the subject's mean removed at fetch time):
- Aging (gaitdb, 5 young vs 5 elderly): elderly mean below young mean, and p < 0.05 on the
  per-recording Welch test. With 5 per group this has far less power than the paper.
  Young mean within 0.15 of 0.87.
- Huntington's (gaitndd, 20 HD vs 16 controls): HD mean below control mean, p < 0.05.
  Control mean within 0.15 of 0.88. Across HD subjects, Spearman correlation between the
  exponent and total functional capacity is positive with p < 0.05.

The subjects on PhysioNet may not be exactly the paper's, so these test direction and
significance, with wide tolerances on the absolute values. The data isn't committed (see
data/fetch_gait.py), so this file is skipped unless the CSVs exist under data/external/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scipy import stats

from complexity_engine.analysis import analyze_dataset

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
AGING = EXTERNAL / "gait_aging.csv"
NDD = EXTERNAL / "gait_ndd.csv"

pytestmark = pytest.mark.skipif(
    not (AGING.exists() and NDD.exists()), reason="Gait data not downloaded; run python data/fetch_gait.py"
)


def _run(path, tmp_path_factory):
    return analyze_dataset(
        path,
        signal_cols=["stride_interval"],
        unit_cols=["subject_id"],
        domain="gait",
        output_dir=tmp_path_factory.mktemp(path.stem),
    )


@pytest.fixture(scope="module")
def aging(tmp_path_factory):
    return _run(AGING, tmp_path_factory)


@pytest.fixture(scope="module")
def ndd(tmp_path_factory):
    return _run(NDD, tmp_path_factory)


def _alpha(result, condition):
    units = result["complexity_by_unit"]
    return units.loc[units["condition"] == condition, "dfa_exponent"]


def _welch(result, condition_a, condition_b):
    rows = result["significance_summary"]
    row = rows[(rows["measure"] == "dfa_exponent") & (rows["condition_a"] == condition_a)
               & (rows["condition_b"] == condition_b)].iloc[0]
    assert row["test"] == "welch_on_recording_values"
    return row


def test_every_subject_has_an_exponent(aging, ndd):
    assert aging["complexity_by_unit"]["dfa_exponent"].notna().all()
    assert ndd["complexity_by_unit"]["dfa_exponent"].notna().all()


def test_young_adults_show_long_range_correlations(aging):
    assert _alpha(aging, "young").mean() == pytest.approx(0.87, abs=0.15)


def test_elderly_exponent_is_lower_than_young(aging):
    row = _welch(aging, "elderly", "young")
    assert (row["n_a"], row["n_b"]) == (5, 5)
    assert row["mean_a"] < row["mean_b"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Elderly 0.751 vs young 0.807 is in the published direction, but p = 0.50 (d = -0.45) with "
        "5 subjects per group. One young subject (y3, 0.654) sits below most elderly ones. The "
        "paper's effect (d ~ 1.3) needs roughly 10 per group for 80% power."
    ),
)
def test_aging_difference_is_significant(aging):
    assert _welch(aging, "elderly", "young")["p_value"] < 0.05


def test_controls_show_long_range_correlations(ndd):
    assert _alpha(ndd, "control").mean() == pytest.approx(0.88, abs=0.15)


def test_huntingtons_exponent_is_lower_and_significant(ndd):
    row = _welch(ndd, "control", "huntington")
    assert (row["n_a"], row["n_b"]) == (16, 20)
    assert row["mean_b"] < row["mean_a"]
    assert row["p_value"] < 0.05


def test_huntingtons_exponent_rises_with_functional_capacity(ndd):
    units = ndd["complexity_by_unit"]
    hd = units[units["condition"] == "huntington"].merge(
        ndd["data"][["subject_id", "severity"]].drop_duplicates(), on="subject_id"
    ).dropna(subset=["severity"])  # one HD subject has no functional-capacity score in the source
    assert len(hd) == 19
    rho, p = stats.spearmanr(hd["severity"], hd["dfa_exponent"])
    assert rho > 0 and p < 0.05
