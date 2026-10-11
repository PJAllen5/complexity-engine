"""Known-answer validation of gap handling, against Ma et al. (2010).

Ma et al. removed stretches of data from long-range correlated signals and joined what was
left. For positively correlated signals the DFA exponent was practically unchanged; for
anticorrelated ones it drifted toward 0.5. This reproduces that setup with fractional
Gaussian noise (65,536 samples, DFA1 over 16-512 samples, 5 realizations), comparing
gap_policy "join" (Segments concatenated) and "segment" against the exponent before loss.

Settings and predictions were fixed before the first run:
- segment stays within 0.05 of the no-loss exponent for alpha = 0.3, 0.5, 0.7, 0.9 at 10%
  and 50% loss (removed stretches averaging 200 samples): reproduced;
- join stays within 0.05 for alpha = 0.7 and 0.9: reproduced;
- join drifts alpha = 0.3 up by at least 0.05 at 50% loss: NOT reproduced (+0.012); see the
  strict xfail. An exploratory rerun found the drift once kept stretches are much shorter
  than the largest window (+0.05 at 20-sample stretches, +0.11 at 5), where segment-aware
  DFA has no windows left and returns NaN. See docs/validation/gap_handling.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from complexity_engine.metrics import Segments, dfa_exponent


def _fgn(n, hurst, rng):
    """Fractional Gaussian noise by circulant embedding (Davies & Harte, 1987)."""
    k = np.arange(n + 1)
    acov = 0.5 * ((k + 1.0) ** (2 * hurst) - 2 * k ** (2.0 * hurst) + np.abs(k - 1.0) ** (2 * hurst))
    row = np.concatenate([acov, acov[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(row).real, 0, None)
    z = rng.standard_normal(row.size) + 1j * rng.standard_normal(row.size)
    return np.fft.fft(np.sqrt(eigenvalues / row.size) * z).real[:n]


def _remove_stretches(x, loss, mean_removed, rng):
    """Alternate kept and removed stretches with exponential lengths; return the kept ones."""
    mean_kept = mean_removed * (1 - loss) / loss
    segments, i, keep = [], 0, bool(rng.integers(2))
    while i < x.size:
        length = max(1, int(rng.exponential(mean_kept if keep else mean_removed)))
        if keep:
            segments.append(x[i : i + length])
        i += length
        keep = not keep
    return [segment for segment in segments if segment.size]


@pytest.fixture(scope="module")
def exponents():
    rng = np.random.default_rng(2010)
    results = {}
    for hurst in (0.3, 0.5, 0.7, 0.9):
        for loss in (0.1, 0.5):
            full, join, segment = [], [], []
            for _ in range(5):
                x = _fgn(65536, hurst, rng)
                kept = _remove_stretches(x, loss, 200, rng)
                full.append(dfa_exponent(x, 16, 512))
                join.append(dfa_exponent(np.concatenate(kept), 16, 512))
                segment.append(dfa_exponent(Segments(kept), 16, 512))
            results[hurst, loss] = (np.mean(full), np.mean(join), np.mean(segment))
    return results


def test_segment_policy_keeps_the_exponent_for_any_correlation(exponents):
    for (hurst, loss), (full, _, segment) in exponents.items():
        assert full == pytest.approx(hurst, abs=0.03), (hurst, loss)
        assert segment == pytest.approx(full, abs=0.05), (hurst, loss)


def test_join_policy_keeps_the_exponent_for_positively_correlated_signals(exponents):
    for hurst in (0.7, 0.9):
        for loss in (0.1, 0.5):
            full, join, _ = exponents[hurst, loss]
            assert join == pytest.approx(full, abs=0.05), (hurst, loss)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Joining shifted alpha = 0.3 by only +0.012 at 50% loss with 200-sample stretches. The "
        "drift Ma et al. (2010) report appears in an exploratory rerun only once stretches are much "
        "shorter than the largest window (+0.05 at 20 samples, +0.11 at 5)."
    ),
)
def test_join_policy_biases_anticorrelated_signals(exponents):
    full, join, _ = exponents[0.3, 0.5]
    assert join - full >= 0.05
