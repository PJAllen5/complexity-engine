"""Synthetic known-answer validation of the neural_activity profile, roadmap item A7
(docs/design/validation-and-new-domains.md).

Calcium imaging reports spikes through an indicator: each spike adds a transient that decays
over a fraction of a second to seconds (Chen et al., 2013). Spike trains are simulated, convolved
with an exponential decay, and given imaging noise. The answers are known by construction.

Criteria, fixed before running:
(a) Poisson spikes (1 Hz) at 10 fps with a 0.5 s decay: under the profile (DFA from 16 frames),
    mean exponent within 0.5 +/- 0.1, since spikes are uncorrelated beyond the decay.
(b) Spikes whose rate follows long-range-correlated noise (fGn, H = 0.9): mean exponent above
    0.7 and at least 0.15 above (a); at least 80% of neurons beat their shuffle surrogates (p < 0.05).
(c) For (a)'s neurons, calcium smoothing lowers sample entropy: every neuron's Delta F/F has lower
    sample entropy than the same spikes plus the same noise without the indicator.
(d) The profile's caution: at 30 fps with a 1.5 s decay (45 frames), min_scale 16 sits inside
    the decay and inflates the exponent above 0.6; min_scale 180 (4 decay times) brings it back
    within 0.5 +/- 0.1.

Result: (b)'s exponent, its surrogates, and (c) pass. (a), (b)'s gap to (a), and (d)'s
min_scale 180 fail as strict xfails, all for one reason: indicator smoothing inflates the DFA
exponent far beyond the decay time. At 3 decay times (16 frames for a 5-frame decay) Poisson
spiking still reads 0.69; at 4 decay times (180 frames for 45) it reads 0.75, and it takes about
10 decay times to approach 0.5. The profile's min_scale of 16 is therefore too small for most
indicators. See docs/validation/neural_activity_synthetic.md.
"""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.metrics import dfa_exponent, sample_entropy, surrogate_test

N_NEURONS, N_FRAMES = 10, 6000


def _fgn(n, hurst, rng):
    """Fractional Gaussian noise by circulant embedding (Davies & Harte, 1987)."""
    k = np.arange(n + 1)
    acov = 0.5 * ((k + 1.0) ** (2 * hurst) - 2 * k ** (2.0 * hurst) + np.abs(k - 1.0) ** (2 * hurst))
    row = np.concatenate([acov, acov[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(row).real, 0, None)
    z = rng.standard_normal(row.size) + 1j * rng.standard_normal(row.size)
    return np.fft.fft(np.sqrt(eigenvalues / row.size) * z).real[:n]


def _neuron(rng, fps, decay_s, correlated=False, rate_hz=1.0, n=N_FRAMES):
    """(Delta F/F, spikes + noise without the indicator) for one simulated neuron."""
    rate = np.full(n, rate_hz / fps)
    if correlated:
        z = _fgn(n, 0.9, rng)
        rate = rate * np.exp(0.8 * (z - z.mean()) / z.std())
    spikes = rng.poisson(rate).astype(float)
    kernel = np.exp(-np.arange(int(8 * decay_s * fps)) / (decay_s * fps))
    noise = 0.05 * rng.standard_normal(n)
    return np.convolve(spikes, kernel)[:n] + noise, spikes + noise


def _recordings(seed, **kwargs):
    rng = np.random.default_rng(seed)
    return [_neuron(rng, **kwargs) for _ in range(N_NEURONS)]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    frames = []
    for condition, correlated, seed in (("poisson", False, 1), ("correlated", True, 2)):
        for i, (dff, _) in enumerate(_recordings(seed, fps=10, decay_s=0.5, correlated=correlated)):
            frames.append(pd.DataFrame({"session": "m1", "neuron_id": f"{condition}{i}", "condition": condition,
                                        "frame": np.arange(N_FRAMES), "dff": dff}))
    path = tmp_path_factory.mktemp("neural") / "dff.csv"
    pd.concat(frames).to_csv(path, index=False)
    result = analyze_dataset(path, signal_cols=["dff"], unit_cols=["session", "neuron_id"], domain="neural_activity",
                             output_dir=path.parent / "out")
    units = result["complexity_by_unit"]
    return {c: units.loc[units["condition"] == c, "dfa_exponent"] for c in ("poisson", "correlated")}


_SMOOTHING = (
    "Indicator smoothing inflates DFA well beyond the decay time: 16 frames is 3.2 decay times "
    "here, and Poisson spiking reads 0.69 (0.51 without the indicator)."
)


@pytest.mark.xfail(strict=True, reason=_SMOOTHING)
def test_poisson_spiking_is_uncorrelated_beyond_the_indicator(pipeline):
    assert pipeline["poisson"].mean() == pytest.approx(0.5, abs=0.1)


def test_correlated_firing_rates_show_long_range_correlation(pipeline):
    assert pipeline["correlated"].mean() > 0.7


@pytest.mark.xfail(strict=True, reason=_SMOOTHING + " So correlated firing (0.81) is only 0.12 above it.")
def test_correlated_firing_stands_out_from_poisson_firing(pipeline):
    assert pipeline["correlated"].mean() - pipeline["poisson"].mean() >= 0.15


def test_correlated_neurons_beat_their_shuffles():
    p_values = []
    for i, (dff, _) in enumerate(_recordings(2, fps=10, decay_s=0.5, correlated=True)):
        observed = dfa_exponent(dff, min_scale=16)
        p_values.append(surrogate_test(dff, "dfa_exponent", observed, {"min_scale": 16}, count=99,
                                       rng=np.random.default_rng(100 + i))["surrogate_p"])
    assert np.mean(np.array(p_values) < 0.05) >= 0.8


def test_calcium_smoothing_lowers_sample_entropy():
    for dff, unsmoothed in _recordings(1, fps=10, decay_s=0.5):
        assert sample_entropy(dff) < sample_entropy(unsmoothed)


@pytest.fixture(scope="module")
def slow_indicator():
    return _recordings(3, fps=30, decay_s=1.5)


def test_min_scale_inside_the_decay_inflates_the_exponent(slow_indicator):
    assert np.mean([dfa_exponent(dff, min_scale=16) for dff, _ in slow_indicator]) > 0.6


@pytest.mark.xfail(
    strict=True,
    reason="4 decay times (180 frames) isn't enough: the exponent is 0.75, and still 0.65 at 400 frames "
    "(9 decay times). The criterion assumed the smoothing's effect ends within a few decay times.",
)
def test_four_decay_times_bring_the_exponent_back_to_half(slow_indicator):
    assert np.mean([dfa_exponent(dff, min_scale=180) for dff, _ in slow_indicator]) == pytest.approx(0.5, abs=0.1)
