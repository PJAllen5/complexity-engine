import numpy as np
import pandas as pd
import pytest
from scipy import stats

from complexity_engine.analysis import (
    analyze_dataset,
    condition_cross_signal_summary,
    condition_significance_summary,
    resolve_signal_columns,
)
from complexity_engine.metrics import compare_conditions, dfa_exponent, mutual_information, sample_entropy, spectral_entropy


def test_sample_entropy_known_value():
    data = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    result = sample_entropy(data, m=2, r=0.25)
    assert np.isfinite(result)
    assert result >= 0.0



def _brute_force_sample_entropy(data, m=2, r=0.2):
    """The original O(n^2) pairwise implementation, kept as a reference for the KD-tree version."""
    data = np.asarray(data, dtype=float)
    tol = r * np.std(data)

    def count_matches(level):
        templates = [tuple(data[i : i + level]) for i in range(len(data) - level + 1)]
        return sum(
            1
            for i in range(len(templates))
            for j in range(i + 1, len(templates))
            if max(abs(templates[i][k] - templates[j][k]) for k in range(level)) <= tol
        )

    b_m, a_m = count_matches(m), count_matches(m + 1)
    return np.nan if b_m == 0 or a_m == 0 else float(-np.log(a_m / b_m))


def _brute_force_dfa_exponent(data, min_scale=8):
    """The original per-window np.polyfit implementation, kept as a reference for the vectorized version."""
    data = np.asarray(data, dtype=float)
    profile = np.cumsum(data - np.mean(data))
    n = len(profile)
    scales = np.unique(np.array([int(s) for s in np.geomspace(min_scale, max(8, n // 4), num=8)], dtype=int))
    scales = scales[(scales >= 2) & (scales < n)]
    rms = []
    for scale in scales:
        residuals = []
        for idx in range(0, n - scale + 1, scale):
            segment = profile[idx : idx + scale]
            x = np.arange(scale)
            residual = segment - np.polyval(np.polyfit(x, segment, 1), x)
            residuals.append(np.sqrt(np.mean(residual**2)))
        rms.append(np.mean(residuals))
    slope, _ = np.polyfit(np.log(scales), np.log(rms), 1)
    return float(slope)


def _regression_series():
    rng = np.random.default_rng(42)
    return {
        "white_noise": rng.standard_normal(600),
        "random_walk": np.cumsum(rng.standard_normal(600)),
        "sine_plus_noise": np.sin(np.linspace(0, 30, 600)) + 0.1 * rng.standard_normal(600),
        # Rounded values put many template distances exactly on the tolerance boundary.
        "rounded_noise_with_ties": np.round(rng.standard_normal(600), 1),
        "short": rng.standard_normal(17),
    }


def test_sample_entropy_matches_brute_force_reference_exactly():
    for name, series in _regression_series().items():
        for r in (0.1, 0.2, 0.5):
            expected = _brute_force_sample_entropy(series, m=2, r=r)
            actual = sample_entropy(series, m=2, r=r)
            assert (np.isnan(expected) and np.isnan(actual)) or actual == expected, (name, r)


def test_dfa_exponent_matches_per_window_polyfit_reference():
    for name, series in _regression_series().items():
        assert dfa_exponent(series) == pytest.approx(_brute_force_dfa_exponent(series), abs=1e-10), name


def test_sample_entropy_handles_long_series_near_theoretical_value():
    # For i.i.d. Gaussian noise, SampEn(m, r) = -ln P(|X - Y| <= r*sigma) with X - Y ~ N(0, 2 sigma^2),
    # independent of m. 100,000 samples was impractical with the original pairwise implementation.
    x = np.random.default_rng(7).standard_normal(100_000)
    expected = -np.log(2 * stats.norm.cdf(0.2 / np.sqrt(2)) - 1)
    assert sample_entropy(x) == pytest.approx(expected, abs=0.02)


def test_spectral_entropy_nonnegative():
    data = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0], dtype=float)
    result = spectral_entropy(data)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


def test_resolve_signal_columns_ignores_metadata():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "condition": ["rest", "task", "rest", "task"],
            "timestamp": [0, 1, 2, 3],
            "signal_a": [1.0, 2.0, 3.0, 4.0],
            "signal_b": [0.5, 1.5, 2.5, 3.5],
            "label": ["x", "x", "y", "y"],
        }
    )
    result = resolve_signal_columns(df)
    assert result == ["signal_a", "signal_b"]


def test_analyze_dataset_supports_multiple_signals(tmp_path):
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"] * 2,
            "condition": ["rest", "rest", "task", "task"] * 2,
            "signal_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "signal_b": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        }
    )
    csv_path = tmp_path / "multi_signal.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(csv_path, signal_cols=["signal_a", "signal_b"], condition_col="condition", output_dir=tmp_path / "outputs")

    assert "complexity_summary" in result
    assert len(result["complexity_summary"]) >= 4
    assert (tmp_path / "outputs" / "report.md").exists()


def test_mutual_information_higher_for_coupled_signals():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y_coupled = x + rng.normal(scale=0.1, size=200)
    y_independent = rng.normal(size=200)

    mi_coupled = mutual_information(x, y_coupled)
    mi_independent = mutual_information(x, y_independent)

    assert mi_coupled > mi_independent


def test_mutual_information_handles_degenerate_input():
    assert mutual_information([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert mutual_information([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_condition_cross_signal_summary_requires_multiple_signals():
    df = pd.DataFrame({"condition": ["rest", "task"], "signal_a": [1.0, 2.0]})
    result = condition_cross_signal_summary(df, ["signal_a"], condition_col="condition")
    assert result.empty


def test_compare_conditions_detects_clear_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=1.0, size=100)
    y = rng.normal(loc=5.0, scale=1.0, size=100)

    result = compare_conditions(x, y)

    assert result["p_value"] < 0.001
    assert result["cohens_d"] < -3.0


def test_compare_conditions_no_difference_not_significant():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=0.0, scale=1.0, size=100)
    y = rng.normal(loc=0.0, scale=1.0, size=100)

    result = compare_conditions(x, y)

    assert result["p_value"] > 0.05


def test_compare_conditions_handles_insufficient_samples():
    result = compare_conditions([1.0], [1.0, 2.0, 3.0])
    assert np.isnan(result["p_value"])
    assert np.isnan(result["cohens_d"])


def test_condition_significance_summary_covers_all_condition_pairs():
    df = pd.DataFrame(
        {
            "condition": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "signal": [1.0, 1.1, 0.9, 1.05, 0.95, 5.0, 5.1, 4.9, 5.05, 4.95, 10.0, 10.1, 9.9, 10.05, 9.95],
        }
    )
    result = condition_significance_summary(df, signal_col="signal", condition_col="condition")

    assert len(result) == 3  # a-b, a-c, b-c
    assert set(zip(result["condition_a"], result["condition_b"])) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_sample_entropy_is_nan_when_too_short():
    assert np.isnan(sample_entropy(np.array([1.0, 2.0, 3.0, 4.0])))


def test_sample_entropy_is_nan_when_no_templates_match():
    # Strictly increasing with a tiny tolerance: no two templates are within r * std.
    assert np.isnan(sample_entropy(np.arange(20, dtype=float) ** 2, r=0.01))


def test_sample_entropy_of_constant_signal_is_zero():
    # Every template matches, so a constant signal is genuinely perfectly regular.
    assert sample_entropy(np.full(20, 3.0)) == 0.0


def test_spectral_entropy_is_nan_when_undefined():
    assert np.isnan(spectral_entropy(np.array([1.0])))
    assert np.isnan(spectral_entropy(np.full(32, 3.0)))


def test_dfa_exponent_is_nan_for_constant_signal():
    assert np.isnan(dfa_exponent(np.full(64, 3.0)))
