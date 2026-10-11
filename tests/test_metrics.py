import numpy as np
import pandas as pd
import pytest
from scipy import signal as sps
from scipy import stats

from complexity_engine.analysis import (
    analyze_dataset,
    condition_cross_signal_summary,
    condition_significance_summary,
    resolve_signal_columns,
    unit_complexity_summary,
)
from complexity_engine.metrics import (
    DEFAULT_METRICS,
    compare_conditions,
    compute_complexity_metrics,
    dfa_exponent,
    effective_sample_size,
    lag1_autocorrelation,
    mutual_information,
    sample_entropy,
    spectral_entropy,
)


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
    """Per-window np.polyfit DFA with Peng et al.'s fluctuation (one RMS over all residuals),
    kept as a reference for the vectorized version."""
    data = np.asarray(data, dtype=float)
    profile = np.cumsum(data - np.mean(data))
    n = len(profile)
    scales = np.unique(np.array([int(s) for s in np.geomspace(min_scale, max(8, n // 4), num=8)], dtype=int))
    scales = scales[(scales >= 2) & (scales < n)]
    fluctuations = []
    for scale in scales:
        squared_residuals = []
        for idx in range(0, n - scale + 1, scale):
            segment = profile[idx : idx + scale]
            x = np.arange(scale)
            squared_residuals.extend((segment - np.polyval(np.polyfit(x, segment, 1), x)) ** 2)
        fluctuations.append(np.sqrt(np.mean(squared_residuals)))
    slope, _ = np.polyfit(np.log(scales), np.log(fluctuations), 1)
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



def test_scale_range_dfa_metrics_on_canonical_signals():
    walk = np.cumsum(np.random.default_rng(10).standard_normal(4000))
    result = compute_complexity_metrics(walk, metrics=["dfa_short_term", "dfa_long_term"])
    assert result["dfa_short_term"] == pytest.approx(1.5, abs=0.1)
    assert result["dfa_long_term"] == pytest.approx(1.5, abs=0.1)


def test_scale_range_dfa_separates_a_crossover():
    # AR(1) is random-walk-like below its correlation time and white-noise-like above it:
    # the two-regime pattern Iyengar et al. (1996) modelled in elderly heart rate. A single
    # exponent over the whole range would blur the two.
    x = sps.lfilter([1.0], [1.0, -0.7], np.random.default_rng(11).standard_normal(4000))
    result = compute_complexity_metrics(x, metrics=["dfa_short_term", "dfa_long_term"])
    assert result["dfa_short_term"] - result["dfa_long_term"] > 0.25


def test_scale_range_dfa_is_nan_when_series_too_short_for_its_range():
    rng = np.random.default_rng(12)
    assert np.isnan(dfa_exponent(rng.standard_normal(255), min_scale=16, max_scale=64))
    assert np.isfinite(dfa_exponent(rng.standard_normal(256), min_scale=16, max_scale=64))


def test_scale_range_dfa_metrics_are_opt_in():
    assert DEFAULT_METRICS == ("sample_entropy", "spectral_entropy", "dfa_exponent")
    assert set(compute_complexity_metrics(np.random.default_rng(13).standard_normal(300))) == set(DEFAULT_METRICS)


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



def _ar1(rng, n, rho):
    return sps.lfilter([1.0], [1.0, -rho], rng.standard_normal(n))


def test_compare_conditions_defaults_to_standard_welch_test():
    rng = np.random.default_rng(3)
    x, y = rng.normal(0.0, 1.0, 40), rng.normal(0.5, 2.0, 25)
    expected = stats.ttest_ind(x, y, equal_var=False)
    result = compare_conditions(x, y)
    assert result["t_statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)


def test_effective_sample_size_shrinks_with_autocorrelation():
    rng = np.random.default_rng(4)
    white = rng.standard_normal(5000)
    persistent = _ar1(rng, 5000, 0.9)
    alternating = np.tile([1.0, -1.0], 50)

    assert effective_sample_size(white) == pytest.approx(5000, rel=0.1)
    # AR(1) with rho = 0.9 is worth about n * 0.1 / 1.9 independent samples.
    assert effective_sample_size(persistent) == pytest.approx(5000 * 0.1 / 1.9, rel=0.25)
    # Negative autocorrelation never credits a series with more than its sample count.
    assert lag1_autocorrelation(alternating) < 0
    assert effective_sample_size(alternating) == 100
    assert effective_sample_size(np.full(10, 2.0)) == 10


def test_autocorrelation_adjusted_test_keeps_false_positive_rate_near_nominal():
    # Two conditions with no real difference, one strongly autocorrelated recording each.
    # Treating every sample as independent rejects most of the time; the adjusted test
    # should reject about 5% of the time.
    rng = np.random.default_rng(5)
    naive_rejections = adjusted_rejections = 0
    trials = 200
    for _ in range(trials):
        a, b = _ar1(rng, 500, 0.9), _ar1(rng, 500, 0.9)
        naive_rejections += compare_conditions(a, b)["p_value"] < 0.05
        adjusted = compare_conditions(a, b, n_x=effective_sample_size(a), n_y=effective_sample_size(b))
        adjusted_rejections += adjusted["p_value"] < 0.05
    assert naive_rejections / trials > 0.4
    assert adjusted_rejections / trials < 0.1


def _subjects_frame(rng, n_subjects=4, n_samples=300, condition_effect=0.0):
    frames = []
    for condition, shift in (("rest", 0.0), ("task", condition_effect)):
        for subject in range(n_subjects):
            offset = rng.normal(0.0, 2.0)  # between-subject differences dominate
            values = offset + shift + _ar1(rng, n_samples, 0.5)
            frames.append(pd.DataFrame({"condition": condition, "subject_id": f"{condition}{subject}", "signal": values}))
    return pd.concat(frames, ignore_index=True)


def test_significance_uses_one_value_per_recording_when_available():
    df = _subjects_frame(np.random.default_rng(6))
    result = condition_significance_summary(df, signal_col="signal", unit_cols=["subject_id"])

    row = result.iloc[0]
    assert row["test"] == "welch_on_recording_values"
    assert row["measure"] == "signal_mean"
    assert (row["n_a"], row["n_b"]) == (4, 4)


def test_recording_level_test_avoids_pseudoreplication_false_positives():
    rng = np.random.default_rng(7)
    naive_rejections = recording_rejections = 0
    trials = 100
    for _ in range(trials):
        df = _subjects_frame(rng)
        naive = compare_conditions(df.loc[df["condition"] == "rest", "signal"], df.loc[df["condition"] == "task", "signal"])
        naive_rejections += naive["p_value"] < 0.05
        result = condition_significance_summary(df, signal_col="signal", unit_cols=["subject_id"])
        recording_rejections += result.iloc[0]["p_value"] < 0.05
    assert naive_rejections / trials > 0.5
    assert recording_rejections / trials < 0.12


def test_significance_tests_complexity_metrics_across_recordings():
    rng = np.random.default_rng(8)
    frames = []
    for condition, rho in (("rest", 0.95), ("task", 0.0)):
        for subject in range(5):
            frames.append(
                pd.DataFrame({"condition": condition, "subject_id": f"s{subject}", "signal": _ar1(rng, 300, rho)})
            )
    df = pd.concat(frames, ignore_index=True)
    unit_summary = unit_complexity_summary(df, "signal", unit_cols=["subject_id"], metrics=["dfa_exponent"])

    result = condition_significance_summary(
        df, signal_col="signal", unit_cols=["subject_id"], unit_summary=unit_summary, metrics=["dfa_exponent"]
    )

    assert set(result["measure"]) == {"signal_mean", "dfa_exponent"}
    dfa_row = result[result["measure"] == "dfa_exponent"].iloc[0]
    assert dfa_row["mean_a"] > dfa_row["mean_b"]
    assert dfa_row["p_value"] < 0.001


def test_significance_falls_back_to_adjusted_samples_with_one_recording_per_condition():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "condition": ["rest"] * 400 + ["task"] * 400,
            "signal": np.concatenate([_ar1(rng, 400, 0.9), _ar1(rng, 400, 0.9)]),
        }
    )
    row = condition_significance_summary(df, signal_col="signal").iloc[0]

    assert row["test"] == "welch_autocorrelation_adjusted"
    assert row["n_a"] < 100 and row["n_b"] < 100


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
