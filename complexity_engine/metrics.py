"""Core signal and complexity metrics used in the MVP workflow."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree


def descriptive_stats(signal: np.ndarray) -> dict[str, float]:
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "variance": np.nan}

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "variance": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy for a one-dimensional time series.

    Returns NaN when sample entropy is undefined: the series is too short to form
    templates, or no template pairs match at length m or m + 1 (-log(0) and 0/0). A
    constant signal returns 0.0, since every template matches -- it is genuinely
    perfectly regular, not a failed computation.
    """
    data = np.asarray(signal, dtype=float)
    if data.size < 2 * m + 1:
        return np.nan

    data_std = np.std(data)
    tol = r * data_std if data_std > 0 else 0.0
    if tol == 0.0:
        return 0.0

    def count_matches(level: int) -> int:
        # Pairs of distinct templates within tol under the Chebyshev (max-abs) distance.
        # A KD-tree counts them without comparing every pair, so run time no longer grows
        # with the square of the series length; the count is identical to a brute-force
        # pairwise comparison (see tests/test_metrics.py).
        templates = np.lib.stride_tricks.sliding_window_view(data, level)
        tree = cKDTree(templates)
        # count_neighbors counts ordered pairs, including each template with itself.
        ordered_pairs = int(tree.count_neighbors(tree, tol, p=np.inf))
        return (ordered_pairs - len(templates)) // 2

    b_m = count_matches(m)
    a_m = count_matches(m + 1)

    if b_m == 0 or a_m == 0:
        return np.nan

    return float(-np.log(a_m / b_m))


def spectral_entropy(signal: np.ndarray) -> float:
    """Compute normalized spectral entropy from the FFT power spectrum.

    Uses squared FFT magnitude (power), not raw magnitude -- spectral entropy is
    defined as the Shannon entropy of the power spectral density (Inouye et al.,
    1991), and using amplitude instead of power was a real bug found by comparing
    against a reference implementation (antropy): for a pure sine wave, amplitude
    gave a normalized entropy around 0.15 while power (correctly) gives ~0.0006,
    matching antropy almost exactly. Squaring makes the distribution more peaked
    around a signal's dominant frequency, which is the whole point of the metric.

    Returns NaN for fewer than 2 samples or a constant signal (no spectral power to
    distribute), rather than 0.0, which would read as "one dominant frequency."
    """
    data = np.asarray(signal, dtype=float)
    if data.size < 2:
        return np.nan

    centered = data - np.mean(data)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[power > 0]
    if power.size == 0:
        return np.nan

    probabilities = power / np.sum(power)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(probabilities.size)
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def _detrended_fluctuation(profile: np.ndarray, scale: int) -> float:
    """DFA fluctuation F(n): RMS residual, over every sample, after removing a least-squares
    line from each non-overlapping window of `scale` samples (a trailing partial window is
    dropped).

    This is Peng et al.'s (1995) definition: one RMS over all residuals, not the mean of
    each window's RMS. The two differ most at small windows, where per-window RMS is noisy
    and averaging it biases the exponent upward (white noise over 4-16 samples gave ~0.64
    instead of the ~0.59 that nolds and this definition give).

    Fits every window at once in closed form rather than calling np.polyfit per window.
    """
    n_windows = len(profile) // scale
    windows = profile[: n_windows * scale].reshape(n_windows, scale)
    x = np.arange(scale, dtype=float)
    x_centered = x - x.mean()
    windows_centered = windows - windows.mean(axis=1, keepdims=True)
    slopes = windows_centered @ x_centered / (x_centered @ x_centered)
    residuals = windows_centered - np.outer(slopes, x_centered)
    return float(np.sqrt(np.mean(residuals**2)))


def dfa_exponent(signal: np.ndarray, min_scale: int = 8, max_scale: int | None = None) -> float:
    """Approximate the detrended fluctuation analysis exponent.

    Fits the log-log slope of fluctuation against window size over 8 log-spaced window
    sizes from min_scale to max_scale samples. Without max_scale, the range runs to a
    quarter of the series length. With max_scale, the series must be at least
    4 * max_scale samples long (so the largest window still averages over four windows);
    otherwise the result is NaN rather than a slope over a silently truncated range.

    Returns NaN for fewer than 16 samples or when fluctuation is zero at some scale
    (e.g. a constant or perfectly linear signal), where the log-log slope is undefined.
    """
    data = np.asarray(signal, dtype=float)
    if data.size < 16:
        return np.nan
    if max_scale is not None and data.size < 4 * max_scale:
        return np.nan

    detrended = data - np.mean(data)
    profile = np.cumsum(detrended)
    n = len(profile)
    upper = max_scale if max_scale is not None else max(8, n // 4)
    scales = np.unique(np.array([int(s) for s in np.geomspace(min_scale, upper, num=8)], dtype=int))
    scales = scales[scales >= 2]
    scales = scales[scales < n]
    if scales.size < 2:
        return np.nan

    rms = [_detrended_fluctuation(profile, int(scale)) for scale in scales]

    if len(rms) < 2 or np.any(np.asarray(rms) <= 0):
        return np.nan

    log_scales = np.log(scales[: len(rms)])
    log_rms = np.log(np.asarray(rms))
    if np.allclose(log_rms, log_rms[0]):
        return 0.0

    slope, _ = np.polyfit(log_scales, log_rms, 1)
    return float(slope)


def lag1_autocorrelation(signal: np.ndarray) -> float:
    """Sample lag-1 autocorrelation; NaN for fewer than 3 samples or a constant signal."""
    data = np.asarray(signal, dtype=float)
    if data.size < 3:
        return np.nan
    centered = data - np.mean(data)
    denominator = np.sum(centered**2)
    if denominator == 0:
        return np.nan
    return float(np.sum(centered[:-1] * centered[1:]) / denominator)


def effective_sample_size(signal: np.ndarray) -> float:
    """Number of independent samples an autocorrelated series is worth for estimating its mean.

    Uses the AR(1) equivalent sample size n(1 - rho) / (1 + rho), with rho the lag-1
    autocorrelation (Zwiers & von Storch, 1995). Negative or undefined rho is treated as 0,
    so a series is never credited with more information than its sample count. The
    correction is approximate: rho estimated from a short series is biased low, and
    long-memory signals are more dependent than AR(1) allows, so the result can still
    overstate the information in strongly persistent series.
    """
    data = np.asarray(signal, dtype=float)
    rho = lag1_autocorrelation(data)
    rho = 0.0 if np.isnan(rho) else max(rho, 0.0)
    return float(data.size * (1.0 - rho) / (1.0 + rho))


def compare_conditions(
    x: np.ndarray, y: np.ndarray, n_x: float | None = None, n_y: float | None = None
) -> dict[str, float]:
    """Compare two samples with Welch's t-test and Cohen's d effect size.

    Welch's t-test doesn't assume equal variance between groups, which suits
    comparing conditions that often differ in variability as well as level.

    n_x / n_y override the number of independent observations used for the standard
    error and degrees of freedom (e.g. an effective sample size for autocorrelated data);
    by default they're the sample counts, which gives the standard Welch's t-test.
    Cohen's d always uses the observed samples, since effect size doesn't depend on
    how much independent evidence there is.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = float(x.size) if n_x is None else float(n_x)
    n_y = float(y.size) if n_y is None else float(n_y)
    nan_result = {"t_statistic": np.nan, "p_value": np.nan, "cohens_d": np.nan}
    if x.size < 2 or y.size < 2 or n_x < 2 or n_y < 2:
        return nan_result

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt(((x.size - 1) * var_x + (y.size - 1) * var_y) / (x.size + y.size - 2))
    cohens_d = float((np.mean(x) - np.mean(y)) / pooled_std) if pooled_std > 0 else np.nan

    se_x, se_y = var_x / n_x, var_y / n_y
    if se_x + se_y == 0:
        return {**nan_result, "cohens_d": cohens_d}
    t_statistic = (np.mean(x) - np.mean(y)) / np.sqrt(se_x + se_y)
    df = (se_x + se_y) ** 2 / (se_x**2 / (n_x - 1) + se_y**2 / (n_y - 1))
    p_value = 2 * stats.t.sf(abs(t_statistic), df)

    return {"t_statistic": float(t_statistic), "p_value": float(p_value), "cohens_d": cohens_d}


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Estimate mutual information (in bits) between two equal-length signals via histogram binning.

    Mutual information quantifies how much knowing one signal reduces uncertainty about the
    other; 0 means the signals appear independent at this binning resolution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4 or x.size != y.size:
        return 0.0
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0

    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)
    outer = np.outer(x_prob, y_prob)

    mask = joint_prob > 0
    return float(np.sum(joint_prob[mask] * np.log2(joint_prob[mask] / outer[mask])))


_METRIC_FUNCTIONS = {
    "sample_entropy": lambda signal: sample_entropy(signal, m=2, r=0.2),
    "spectral_entropy": spectral_entropy,
    "dfa_exponent": dfa_exponent,
    # Separate short- and long-range exponents, for signals whose correlation structure
    # changes with scale. The ranges are the heart-rate convention (Peng et al., 1995),
    # in samples (beats for RR intervals); other domains may need different ranges.
    "dfa_short_term": lambda signal: dfa_exponent(signal, min_scale=4, max_scale=16),
    "dfa_long_term": lambda signal: dfa_exponent(signal, min_scale=16, max_scale=64),
}


# Computed when a domain profile doesn't name its metrics; the scale-range DFA variants
# are opt-in, since they're only meaningful for signals with a known crossover scale.
DEFAULT_METRICS = ("sample_entropy", "spectral_entropy", "dfa_exponent")


def compute_complexity_metrics(signal: np.ndarray, metrics: Sequence[str] | None = None) -> dict[str, float]:
    """Compute complexity metrics for a signal.

    If `metrics` is omitted, DEFAULT_METRICS are computed. Pass an explicit
    subset (e.g. a domain profile's `recommended_metrics`) to compute only those.
    """
    signal = np.asarray(signal, dtype=float)
    selected = tuple(metrics) if metrics else DEFAULT_METRICS
    unknown = [name for name in selected if name not in _METRIC_FUNCTIONS]
    if unknown:
        supported = ", ".join(sorted(_METRIC_FUNCTIONS))
        raise ValueError(f"Unknown complexity metric(s): {', '.join(unknown)}. Supported metrics: {supported}.")
    return {name: float(_METRIC_FUNCTIONS[name](signal)) for name in selected}
