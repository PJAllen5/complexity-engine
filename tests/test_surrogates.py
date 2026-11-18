"""Surrogate baselines (roadmap item E2, docs/design/validation-and-new-domains.md).

Known answers, fixed before running:
- Fractional Gaussian noise with Hurst exponent 0.8 has a DFA exponent near 0.8. Shuffling
  destroys its correlations, so shuffle surrogates sit near 0.5 and the observed value is far
  outside them (p < 0.05). Phase randomization keeps the power spectrum, so phase surrogates
  keep the exponent: the observed value is inside them (p > 0.05).
- White noise has no temporal structure, so against shuffle surrogates p < 0.05 should occur
  about 5% of the time: at most 6 of 40 series (P(X >= 7) ~ 0.01 at 5%).
- A sine wave is far more regular than its shuffled values: lower sample entropy, p < 0.05.
"""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.domain_profiles import DomainProfile
from complexity_engine.metrics import Segments, dfa_exponent, make_surrogate, sample_entropy, surrogate_test


def _fgn(n, hurst, rng):
    """Fractional Gaussian noise by circulant embedding (Davies & Harte, 1987)."""
    k = np.arange(n + 1)
    acov = 0.5 * ((k + 1.0) ** (2 * hurst) - 2 * k ** (2.0 * hurst) + np.abs(k - 1.0) ** (2 * hurst))
    row = np.concatenate([acov, acov[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(row).real, 0, None)
    z = rng.standard_normal(row.size) + 1j * rng.standard_normal(row.size)
    return np.fft.fft(np.sqrt(eigenvalues / row.size) * z).real[:n]


DFA = {"min_scale": 16}


@pytest.fixture(scope="module")
def correlated():
    x = _fgn(4096, 0.8, np.random.default_rng(1))
    return x, dfa_exponent(x, **DFA)


def test_shuffle_surrogates_destroy_long_range_correlation(correlated):
    x, observed = correlated
    result = surrogate_test(x, "dfa_exponent", observed, DFA, kind="shuffle", count=99, rng=np.random.default_rng(2))
    assert result["surrogate_mean"] == pytest.approx(0.5, abs=0.05)
    assert observed - result["surrogate_mean"] > 0.2
    assert result["surrogate_p"] < 0.05


def test_phase_surrogates_keep_linear_correlation(correlated):
    x, observed = correlated
    result = surrogate_test(x, "dfa_exponent", observed, DFA, kind="phase", count=99, rng=np.random.default_rng(3))
    assert result["surrogate_mean"] == pytest.approx(observed, abs=0.05)
    assert result["surrogate_p"] > 0.05


def test_white_noise_rarely_differs_from_its_shuffles():
    rng = np.random.default_rng(4)
    p_values = []
    for _ in range(40):
        x = rng.standard_normal(1024)
        result = surrogate_test(x, "dfa_exponent", dfa_exponent(x, **DFA), DFA, count=49, rng=rng)
        p_values.append(result["surrogate_p"])
    assert sum(p < 0.05 for p in p_values) <= 6


def test_a_sine_is_more_regular_than_its_shuffles():
    x = np.sin(np.linspace(0, 40 * np.pi, 1000))
    result = surrogate_test(x, "sample_entropy", sample_entropy(x), count=49, rng=np.random.default_rng(5))
    assert sample_entropy(x) < result["surrogate_mean"] and result["surrogate_p"] < 0.05


def test_shuffle_keeps_values_and_segment_lengths():
    signal = Segments([np.arange(5.0), np.arange(5.0, 12.0)])
    surrogate = make_surrogate(signal, "shuffle", np.random.default_rng(6))
    assert isinstance(surrogate, Segments) and [len(s) for s in surrogate] == [5, 7]
    assert sorted(np.concatenate(surrogate)) == list(np.arange(12.0))


def test_phase_surrogate_keeps_each_segments_spectrum():
    rng = np.random.default_rng(7)
    signal = Segments([rng.standard_normal(64), rng.standard_normal(101)])
    surrogate = make_surrogate(signal, "phase", np.random.default_rng(8))
    for original, copy in zip(signal, surrogate):
        assert np.allclose(np.abs(np.fft.rfft(original - original.mean())), np.abs(np.fft.rfft(copy - copy.mean())))
        assert copy.mean() == pytest.approx(original.mean())


def test_surrogates_are_reproducible_from_the_seed(correlated):
    x, observed = correlated
    runs = [surrogate_test(x, "dfa_exponent", observed, DFA, count=19, rng=np.random.default_rng(seed))
            for seed in (9, 9, 10)]
    assert runs[0] == runs[1] and runs[0] != runs[2]


def test_nan_observation_gives_nan_p():
    assert np.isnan(surrogate_test(np.ones(10), "dfa_exponent", np.nan, count=19)["surrogate_p"])


@pytest.mark.parametrize(
    "surrogates, message",
    [
        ({"type": "iaaft"}, "type must be one of shuffle, phase"),
        ({"count": 10}, "p can never reach 0.05"),
        ({"seed": -1}, "seed must be a whole number"),
        ({"n": 5}, "surrogates must be a mapping"),
        (["shuffle"], "surrogates must be a mapping"),
    ],
)
def test_profile_rejects_invalid_surrogate_settings(surrogates, message):
    with pytest.raises(ValueError, match=message):
        DomainProfile.from_mapping({"name": "t", "surrogates": surrogates})


def test_profile_fills_surrogate_defaults():
    assert DomainProfile.from_mapping({"name": "t", "surrogates": {"type": "phase"}}).surrogates == {
        "type": "phase", "count": 200, "seed": 0,
    }
    assert DomainProfile.from_mapping({"name": "t"}).surrogates is None


def _write(tmp_path, name, surrogates_yaml=""):
    rng = np.random.default_rng(11)
    frames = [
        pd.DataFrame({"subject_id": f"s{i}", "condition": condition, "signal": _fgn(1024, hurst, rng)})
        for i, (condition, hurst) in enumerate([("noise", 0.5), ("noise", 0.5), ("persistent", 0.85), ("persistent", 0.85)])
    ]
    data = tmp_path / "data.csv"
    pd.concat(frames).to_csv(data, index=False)
    profile = tmp_path / f"{name}.yaml"
    profile.write_text(
        f"{name}:\n  name: {name}\n  description: Test.\n  recommended_metrics: [dfa_exponent]\n"
        f"  metric_parameters:\n    dfa_exponent: {{min_scale: 16}}\n{surrogates_yaml}",
        encoding="utf-8",
    )
    return data, profile


def test_pipeline_reports_surrogate_baselines(tmp_path):
    data, profile = _write(tmp_path, "with_surrogates", "  surrogates: {count: 39, seed: 1}\n")
    result = analyze_dataset(data, signal_cols=["signal"], domain="with_surrogates", domain_config=profile,
                             output_dir=tmp_path / "out")
    units = result["complexity_by_unit"].set_index("subject_id")
    assert units.loc[["s2", "s3"], "dfa_exponent_surrogate_p"].max() < 0.05
    assert units["dfa_exponent_surrogate_mean"].between(0.4, 0.6).all()
    assert "dfa_exponent_surrogate_mean" in result["complexity_summary"].columns
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Surrogate baselines" in report and "39 shuffled copies" in report
    assert "signal, persistent, dfa_exponent:" in report and "2 of 2 recording(s) differ at p < 0.05" in report


def test_profiles_without_surrogates_keep_their_columns(tmp_path):
    data, profile = _write(tmp_path, "without_surrogates")
    result = analyze_dataset(data, signal_cols=["signal"], domain="without_surrogates", domain_config=profile,
                             output_dir=tmp_path / "out")
    assert not any("surrogate" in c for c in result["complexity_by_unit"].columns)
    assert "## Surrogate baselines" not in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
