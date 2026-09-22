"""Core signal and complexity metrics used in the MVP workflow."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


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
    """Compute sample entropy for a one-dimensional time series."""
    data = np.asarray(signal, dtype=float)
    if data.size < 2 * m + 1:
        return 0.0

    data_std = np.std(data)
    tol = r * data_std if data_std > 0 else 0.0
    if tol == 0.0:
        return 0.0

    def count_matches(level: int) -> int:
        templates = [tuple(data[i : i + level]) for i in range(len(data) - level + 1)]
        matches = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if max(abs(templates[i][k] - templates[j][k]) for k in range(level)) <= tol:
                    matches += 1
        return matches

    b_m = count_matches(m)
    a_m = count_matches(m + 1)

    if b_m == 0:
        return 0.0
    if a_m == 0:
        return 0.0

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
    """
    data = np.asarray(signal, dtype=float)
    if data.size < 2:
        return 0.0

    centered = data - np.mean(data)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[power > 0]
    if power.size == 0:
        return 0.0

    probabilities = power / np.sum(power)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(probabilities.size)
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def dfa_exponent(signal: np.ndarray, min_scale: int = 8) -> float:
    """Approximate the detrended fluctuation analysis exponent."""
    data = np.asarray(signal, dtype=float)
    if data.size < 16:
        return np.nan

    detrended = data - np.mean(data)
    profile = np.cumsum(detrended)
    n = len(profile)
    scales = np.unique(np.array([int(s) for s in np.geomspace(min_scale, max(8, n // 4), num=8)], dtype=int))
    scales = scales[scales >= 2]
    scales = scales[scales < n]
    if scales.size < 2:
        return np.nan

    rms = []
    for scale in scales:
        residuals = []
        for idx in range(0, n - scale + 1, scale):
            segment = profile[idx : idx + scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend
            residuals.append(np.sqrt(np.mean(residual**2)))
        if residuals:
            rms.append(np.mean(residuals))

    if len(rms) < 2:
        return np.nan

    log_scales = np.log(scales[: len(rms)])
    log_rms = np.log(np.asarray(rms))
    if np.allclose(log_rms, log_rms[0]):
        return 0.0

    slope, _ = np.polyfit(log_scales, log_rms, 1)
    return float(slope)


def compare_conditions(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compare two samples with Welch's t-test and Cohen's d effect size.

    Welch's t-test doesn't assume equal variance between groups, which suits
    comparing conditions that often differ in variability as well as level.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return {"t_statistic": np.nan, "p_value": np.nan, "cohens_d": np.nan}

    t_statistic, p_value = stats.ttest_ind(x, y, equal_var=False)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt(((x.size - 1) * var_x + (y.size - 1) * var_y) / (x.size + y.size - 2))
    cohens_d = float((np.mean(x) - np.mean(y)) / pooled_std) if pooled_std > 0 else np.nan

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
}


def compute_complexity_metrics(signal: np.ndarray, metrics: Sequence[str] | None = None) -> dict[str, float]:
    """Compute complexity metrics for a signal.

    If `metrics` is omitted, every supported metric is computed. Pass an explicit
    subset (e.g. a domain profile's `recommended_metrics`) to compute only those.
    """
    signal = np.asarray(signal, dtype=float)
    selected = tuple(metrics) if metrics else tuple(_METRIC_FUNCTIONS)
    unknown = [name for name in selected if name not in _METRIC_FUNCTIONS]
    if unknown:
        supported = ", ".join(sorted(_METRIC_FUNCTIONS))
        raise ValueError(f"Unknown complexity metric(s): {', '.join(unknown)}. Supported metrics: {supported}.")
    return {name: float(_METRIC_FUNCTIONS[name](signal)) for name in selected}
