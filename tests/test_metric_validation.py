"""Validates the hand-rolled complexity metrics in metrics.py against independent
reference implementations and known theoretical values.

Nothing else in this project's test suite checks whether these numbers are actually
correct -- every other test only checks qualitative properties (e.g. "higher for a
more irregular signal"), which would never catch a metric that's internally
consistent but wrong relative to the literature it's supposed to implement. This
file exists because that gap turned out to matter: comparing spectral_entropy
against antropy's reference implementation found a real bug (see the dedicated test
below) that the rest of the suite had never caught.

Requires the validation-only dependencies in requirements-dev.txt (antropy, nolds,
scikit-learn) -- these are not part of the pipeline's own runtime requirements, so
this whole file is skipped automatically if they aren't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from complexity_engine.metrics import dfa_exponent, mutual_information, sample_entropy, spectral_entropy

antropy = pytest.importorskip("antropy")
nolds = pytest.importorskip("nolds")
sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _white_noise(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


def _sine(n: int = 1000, cycles: int = 10) -> np.ndarray:
    t = np.linspace(0, cycles * 2 * np.pi, n)
    return np.sin(t)


def _random_walk(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


# --- sample_entropy -----------------------------------------------------------
# Directly comparable: antropy's defaults (order=2, tolerance=0.2*std, Chebyshev
# distance) match our own (m=2, r=0.2, max-norm) exactly, so these should agree
# closely rather than just qualitatively.


def test_sample_entropy_matches_antropy_on_white_noise():
    x = _white_noise()
    assert sample_entropy(x) == pytest.approx(antropy.sample_entropy(x), rel=0.05)


def test_sample_entropy_matches_antropy_on_sine_wave():
    x = _sine()
    assert sample_entropy(x) == pytest.approx(antropy.sample_entropy(x), rel=0.05)


# --- spectral_entropy -----------------------------------------------------------


def test_spectral_entropy_matches_antropy_on_white_noise():
    x = _white_noise()
    theirs = antropy.spectral_entropy(x, sf=1.0, method="fft", normalize=True)
    assert spectral_entropy(x) == pytest.approx(theirs, abs=0.02)


def test_spectral_entropy_matches_antropy_on_sine_wave():
    """Regression test for a real bug found by this validation work: spectral_entropy
    used to compute the Shannon entropy of the FFT *amplitude* spectrum instead of
    the *power* spectrum (squared amplitude) -- amplitude instead of power is not
    the standard definition (Inouye et al., 1991; the metric's own docstring already
    claimed "power spectrum" before the code matched that claim). For a pure sine
    wave, essentially all power should concentrate in one frequency bin, giving a
    near-zero normalized entropy; the buggy amplitude-based version gave ~0.15
    instead, off by more than two orders of magnitude from the correct value.
    """
    x = _sine()
    theirs = antropy.spectral_entropy(x, sf=1.0, method="fft", normalize=True)
    ours = spectral_entropy(x)
    assert theirs < 0.01, "sanity check: antropy itself should call a pure tone near-zero entropy"
    assert ours < 0.01
    assert ours == pytest.approx(theirs, abs=0.01)


# --- dfa_exponent -----------------------------------------------------------
# DFA has a well-known theoretical expectation for these two canonical signals:
# white noise (no long-range correlation) -> ~0.5; a random walk (integrated white
# noise, maximal long-range correlation) -> ~1.5. A single trial has substantial
# sampling variance (confirmed empirically against antropy/nolds, std ~0.03-0.065
# even at n=2000), so these average over several seeds rather than checking one.


def test_dfa_exponent_white_noise_is_near_theoretical_value():
    values = [dfa_exponent(_white_noise(n=2000, seed=s)) for s in range(10)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.1)


def test_dfa_exponent_random_walk_is_near_theoretical_value():
    values = [dfa_exponent(_random_walk(n=2000, seed=s)) for s in range(10)]
    assert np.mean(values) == pytest.approx(1.5, abs=0.1)


def test_dfa_exponent_matches_reference_implementations_on_average():
    seeds = range(10)
    ours = np.mean([dfa_exponent(_white_noise(n=2000, seed=s)) for s in seeds])
    theirs_antropy = np.mean([antropy.detrended_fluctuation(_white_noise(n=2000, seed=s)) for s in seeds])
    theirs_nolds = np.mean([nolds.dfa(_white_noise(n=2000, seed=s)) for s in seeds])
    assert ours == pytest.approx(theirs_antropy, abs=0.1)
    assert ours == pytest.approx(theirs_nolds, abs=0.1)


# --- mutual_information -----------------------------------------------------------
# No off-the-shelf library computes histogram-based MI between two continuous
# signals the way we do, so this cross-checks the underlying discrete MI formula
# against sklearn's well-established implementation by discretizing both signals
# with our own bin edges and feeding the resulting labels to sklearn directly --
# an exact match confirms the formula itself is implemented correctly.


def test_mutual_information_matches_sklearn_with_matched_binning():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = x + rng.normal(scale=0.3, size=500)
    bins = 8

    ours = mutual_information(x, y, bins=bins)

    _, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_binned = np.digitize(x, x_edges[1:-1])
    y_binned = np.digitize(y, y_edges[1:-1])
    theirs_bits = sklearn_metrics.mutual_info_score(x_binned, y_binned) / np.log(2)

    assert ours == pytest.approx(theirs_bits, abs=1e-9)
