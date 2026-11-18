"""Known-answer validation of the stress profile on WESAD, roadmap item A4
(docs/design/validation-and-new-domains.md).

Known answer (directional): the Trier Social Stress Test (Kirschbaum et al., 1993) reliably raises
heart rate and skin conductance. This validates the multi-recording condition comparison on real
physiology, not a published complexity value; it is the roadmap's weakest known answer.

Criteria, fixed before any statistic was computed on the derived data:
1. 15 subjects, each with a baseline and a stress segment.
2. R-peak sanity (no reference annotations exist): at least 95% of RR intervals within 0.4-1.5 s.
3. Mean heart rate higher under stress, per-recording Welch p < 0.05.
4. Mean skin conductance higher under stress, per-recording Welch p < 0.05.
Exploratory, stated in advance but not pass/fail: heart-rate sample entropy lower under stress.

The data isn't committed and needs a manual download (see data/fetch_wesad.py), so this file is
skipped unless the derived CSVs exist under data/external/.

Result: 1-3 pass (heart rate 73 -> 99 bpm, all 15 subjects, p < 0.001; 99.6% of RR intervals
plausible); 4 fails (strict xfail, p = 0.137 unpaired). Exploratory: heart-rate sample entropy
falls under stress (1.38 -> 0.99, p = 0.004). See docs/validation/wesad.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
HEART, EDA = EXTERNAL / "wesad_heart.csv", EXTERNAL / "wesad_eda.csv"

pytestmark = pytest.mark.skipif(not (HEART.exists() and EDA.exists()),
                                reason="WESAD not prepared; see data/fetch_wesad.py")


def _run(path, signal, tmp_path_factory):
    return analyze_dataset(path, signal_cols=[signal], unit_cols=["subject_id"], domain="stress",
                           output_dir=tmp_path_factory.mktemp(signal))


def _level(result):
    rows = result["significance_summary"]
    row = rows[rows["measure"] == "signal_mean"].iloc[0]
    assert (row["condition_a"], row["condition_b"], row["n_a"], row["n_b"]) == ("baseline", "stress", 15, 15)
    return row


@pytest.fixture(scope="module")
def heart(tmp_path_factory):
    return _run(HEART, "heart_rate", tmp_path_factory)


@pytest.fixture(scope="module")
def eda(tmp_path_factory):
    return _run(EDA, "eda_us", tmp_path_factory)


def test_every_subject_has_both_conditions():
    data = pd.read_csv(HEART)
    assert data["subject_id"].nunique() == 15
    assert (data.groupby("subject_id")["condition"].nunique() == 2).all()


def test_r_peak_detection_is_plausible():
    rr = pd.read_csv(HEART)["rr_interval"]
    assert rr.between(0.4, 1.5).mean() >= 0.95


def test_heart_rate_rises_under_stress(heart):
    row = _level(heart)
    assert row["mean_b"] > row["mean_a"] and row["p_value"] < 0.05


@pytest.mark.xfail(
    strict=True,
    reason="Skin conductance rises under stress (3.91 -> 5.96 uS, 13 of 15 subjects), but the engine's "
    "per-recording test treats the same subjects' two recordings as unpaired, and resting skin conductance "
    "varies so much between people that p = 0.137.",
)
def test_skin_conductance_rises_under_stress(eda):
    row = _level(eda)
    assert row["mean_b"] > row["mean_a"] and row["p_value"] < 0.05
