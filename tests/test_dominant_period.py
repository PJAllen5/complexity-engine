"""dominant_period / dominant_power_share (roadmap E5) and per-metric min_samples (roadmap E4),
docs/design/validation-and-new-domains.md.

Known answers, fixed before running:
- A sine with period 10 in noise (n = 1000): dominant_period 10 (exactly, since 1000/100 is
  a periodogram bin), power share above 0.5.
- Two sines, periods 25 (amplitude 2) and 7 (amplitude 1): dominant_period 25.
- White noise (n = 1000): power share below 0.05 (the expected share of one bin is ~2/n).
- A steep linear trend plus a period-20 sine: with detrend="linear", dominant_period 20.
- A period longer than half the series is never reported.
"""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.metrics import (
    Segments,
    compute_complexity_metrics,
    dfa_exponent,
    dominant_period,
    dominant_power_share,
    sample_entropy,
    spectral_entropy,
)

T = np.arange(1000)


def test_a_noisy_sine_has_its_period():
    x = np.sin(2 * np.pi * T / 10) + 0.5 * np.random.default_rng(1).standard_normal(T.size)
    assert dominant_period(x) == pytest.approx(10.0)
    assert dominant_power_share(x) > 0.5


def test_the_stronger_of_two_rhythms_wins():
    x = 2 * np.sin(2 * np.pi * T / 25) + np.sin(2 * np.pi * T / 7)
    assert dominant_period(x) == pytest.approx(25.0)


def test_white_noise_has_no_dominant_rhythm():
    assert dominant_power_share(np.random.default_rng(2).standard_normal(1000)) < 0.05


def test_linear_detrending_uncovers_a_rhythm_under_a_trend():
    x = 0.05 * T + np.sin(2 * np.pi * T / 20)
    assert dominant_period(x, detrend="linear") == pytest.approx(20.0)


def test_periods_longer_than_half_the_series_are_never_reported():
    x = np.sin(2 * np.pi * np.arange(100) / 80)  # under two cycles in 100 samples
    assert dominant_period(x) <= 50


def test_degenerate_input_is_nan():
    assert np.isnan(dominant_period(np.ones(50))) and np.isnan(dominant_period(np.arange(3.0)))
    with pytest.raises(ValueError, match="detrend must be"):
        dominant_period(np.arange(10.0), detrend="quadratic")


def test_segments_use_the_longest_segment():
    long = np.sin(2 * np.pi * np.arange(400) / 8)
    assert dominant_period(Segments([np.random.default_rng(3).standard_normal(50), long])) == pytest.approx(8.0)


@pytest.mark.parametrize("metric", [sample_entropy, spectral_entropy, dfa_exponent, dominant_period])
def test_min_samples_makes_short_recordings_nan(metric):
    x = np.random.default_rng(4).standard_normal(300)
    assert np.isnan(metric(x, min_samples=301))
    assert np.isfinite(metric(x, min_samples=300))
    assert np.isfinite(metric(x))


def test_min_samples_counts_all_segments():
    x = np.random.default_rng(5).standard_normal(200)
    assert np.isfinite(sample_entropy(Segments([x[:100], x[100:]]), min_samples=200))


def test_min_samples_must_be_a_positive_whole_number():
    with pytest.raises(ValueError, match="min_samples must be a positive whole number"):
        sample_entropy(np.arange(10.0), min_samples=0)


def test_new_metrics_are_opt_in():
    assert "dominant_period" not in compute_complexity_metrics(np.random.default_rng(6).standard_normal(100))


def test_pipeline_reports_short_recordings_and_dominant_period(tmp_path):
    rng = np.random.default_rng(7)
    frames = [
        pd.DataFrame({"subject_id": unit, "condition": cond,
                      "signal": np.sin(2 * np.pi * np.arange(n) / 12) + 0.3 * rng.standard_normal(n)})
        for unit, cond, n in [("s1", "a", 600), ("s2", "a", 60), ("s3", "b", 600), ("s4", "b", 600)]
    ]
    data = tmp_path / "data.csv"
    pd.concat(frames).to_csv(data, index=False)
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "rhythm:\n  name: rhythm\n  description: Test.\n"
        "  recommended_metrics: [sample_entropy, dominant_period, dominant_power_share]\n"
        "  metric_parameters:\n    sample_entropy: {min_samples: 100}\n", encoding="utf-8")
    result = analyze_dataset(data, signal_cols=["signal"], domain="rhythm", domain_config=profile,
                             output_dir=tmp_path / "out")
    units = result["complexity_by_unit"].set_index("subject_id")
    assert np.isnan(units.loc["s2", "sample_entropy"]) and units.drop("s2")["sample_entropy"].notna().all()
    assert units.loc["s1", "dominant_period"] == pytest.approx(12.0)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "sample_entropy(m=2, r=0.2, min_samples=100)" in report
    assert ("sample_entropy is not computed for 1 of 4 recording(s) of signal shorter than the profile's "
            "min_samples (100; shortest has 60)") in report
