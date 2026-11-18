"""Signal transforms (roadmap E1, docs/design/validation-and-new-domains.md).

Known answers, fixed before running:
- diff / log_return / abs / zscore give their textbook values, within each recording only.
- bandpass(15, 25) at 100 Hz keeps a 20 Hz sine (correlation > 0.99 away from the edges) and
  removes a 2 Hz one.
- envelope of a 10 Hz carrier amplitude-modulated at 0.5 Hz recovers the modulation
  (correlation > 0.99 away from the edges).
- Log returns of a geometric random walk are uncorrelated: DFA exponent within 0.5 +/- 0.1.
"""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset, apply_signal_transforms
from complexity_engine.domain_profiles import DomainProfile
from complexity_engine.transforms import apply_steps, parse_step, parse_transforms

FS = 100.0
T = np.arange(2000) / FS


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"x": ["smooth"]}, "unknown transform step 'smooth'"),
        ({"x": ["bandpass(8)"]}, "takes 2 argument"),
        ({"x": ["bandpass(13, 8)"]}, "0 < low < high"),
        ({"x": ["bandpass(a, b)"]}, "non-numeric"),
        ({"x": []}, "has no steps"),
        ({"x": {"from": "y"}}, "must be a list of steps"),
        (["diff"], "transforms must map"),
    ],
)
def test_invalid_transforms_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_transforms(raw)


def test_profile_rejects_bad_settings_with_its_name():
    with pytest.raises(ValueError, match="Domain profile 't': transforms.x: unknown transform step"):
        DomainProfile.from_mapping({"name": "t", "transforms": {"x": ["nope"]}})
    with pytest.raises(ValueError, match="sampling_interval must be a positive duration"):
        DomainProfile.from_mapping({"name": "t", "sampling_interval": "160Hz"})


def test_parse_step():
    assert parse_step(" bandpass( 8, 13 ) ") == ("bandpass", (8.0, 13.0))
    assert parse_step("abs") == ("abs", ())


def test_elementwise_steps():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert np.allclose(apply_steps(x, [("diff", ())])[1:], [1, 2, 4]) and np.isnan(apply_steps(x, [("diff", ())])[0])
    assert np.allclose(apply_steps(x, [("log_return", ())])[1:], np.log(2))
    assert np.allclose(apply_steps(x, [("log_return", ()), ("diff", ())])[2:], 0) and np.isnan(
        apply_steps(x, [("log_return", ()), ("diff", ())])[:2]).all()
    assert np.allclose(apply_steps(np.array([-1.0, 2.0]), [("abs", ())]), [1, 2])
    z = apply_steps(np.arange(10.0), [("zscore", ())])
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    with pytest.raises(ValueError, match="strictly positive"):
        apply_steps(np.array([1.0, 0.0]), [("log_return", ())])


def test_bandpass_keeps_the_band_and_removes_the_rest():
    slow, fast = np.sin(2 * np.pi * 2 * T), np.sin(2 * np.pi * 20 * T)
    filtered = apply_steps(slow + fast, [("bandpass", (15, 25))], FS)
    inner = slice(200, -200)
    assert np.corrcoef(filtered[inner], fast[inner])[0, 1] > 0.99
    assert abs(np.corrcoef(filtered[inner], slow[inner])[0, 1]) < 0.05
    with pytest.raises(ValueError, match="below half the sampling rate"):
        apply_steps(fast, [("bandpass", (15, 60))], FS)
    with pytest.raises(ValueError, match="needs a sampling rate"):
        apply_steps(fast, [("bandpass", (15, 25))], None)


def test_envelope_recovers_amplitude_modulation():
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 0.5 * T)
    envelope = apply_steps(modulation * np.sin(2 * np.pi * 10 * T), [("envelope", ())])
    inner = slice(200, -200)
    assert np.corrcoef(envelope[inner], modulation[inner])[0, 1] > 0.99


def _two_units(values_a, values_b, extra=None):
    frame = pd.DataFrame({
        "condition": "c", "subject_id": ["a"] * len(values_a) + ["b"] * len(values_b),
        "time": list(range(len(values_a))) + list(range(len(values_b))),
        "price": np.concatenate([values_a, values_b]),
    })
    return frame.assign(**(extra or {}))


def test_transforms_never_cross_recordings_and_drop_first_samples():
    df = _two_units(np.array([1.0, 2.0, 4.0]), np.array([100.0, 50.0, 25.0]))
    transforms = parse_transforms({"returns": {"from": "price", "steps": ["log_return"]}})
    out, signals, info = apply_signal_transforms(df, transforms, ["price"], unit_cols=["subject_id"], time_col="time")
    assert signals == ["returns"] and info["rows_dropped"] == 2
    assert np.allclose(out.loc[out["subject_id"] == "a", "returns"], np.log(2))
    assert np.allclose(out.loc[out["subject_id"] == "b", "returns"], np.log(0.5))


def test_transforms_follow_time_order_not_file_order():
    df = _two_units(np.array([4.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    df.loc[df["subject_id"] == "a", "time"] = [2, 0, 1]  # in time order the prices are 1, 2, 4
    out, _, _ = apply_signal_transforms(df, parse_transforms({"price": ["diff"]}), ["price"],
                                        unit_cols=["subject_id"], time_col="time")
    a = out[out["subject_id"] == "a"].sort_values("time")
    assert list(a["price"]) == [1.0, 2.0]


def test_bandpass_needs_a_rate_and_takes_it_from_the_profile():
    df = _two_units(np.sin(np.arange(500.0)), np.sin(np.arange(500.0)))
    transforms = parse_transforms({"price": ["bandpass(5, 15)"]})
    with pytest.raises(ValueError, match="sampling_interval"):
        apply_signal_transforms(df, transforms, ["price"], unit_cols=["subject_id"], time_col="time")
    out, _, info = apply_signal_transforms(df, transforms, ["price"], unit_cols=["subject_id"], time_col="time",
                                           sampling_interval="0.01s")
    assert info["rows_dropped"] == 0 and out["price"].notna().all()


def test_pipeline_analyzes_returns_and_volatility_from_prices(tmp_path):
    rng = np.random.default_rng(1)
    frames = []
    for unit, condition in [("s1", "a"), ("s2", "a"), ("s3", "b"), ("s4", "b")]:
        price = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(2000)))
        frames.append(pd.DataFrame({"subject_id": unit, "condition": condition, "time": np.arange(2000), "price": price}))
    data = tmp_path / "prices.csv"
    pd.concat(frames).to_csv(data, index=False)
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "markets:\n  name: markets\n  description: Test.\n  recommended_metrics: [dfa_exponent]\n"
        "  transforms:\n    returns: {from: price, steps: [log_return]}\n"
        "    volatility: {from: price, steps: [log_return, abs]}\n", encoding="utf-8")
    result = analyze_dataset(data, signal_cols=["price"], domain="markets", domain_config=profile,
                             output_dir=tmp_path / "out")
    assert result["selected_signals"] == ["returns", "volatility"]
    units = result["complexity_by_unit"]
    returns = units.loc[units["signal"] == "returns", "dfa_exponent"]
    assert returns.mean() == pytest.approx(0.5, abs=0.1)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Signal transforms" in report
    assert "- price -> returns: log_return" in report and "- price -> volatility: log_return, abs" in report
    assert "4 row(s) dropped" in report


def test_profiles_without_transforms_are_unchanged(tmp_path):
    data = tmp_path / "d.csv"
    _two_units(np.arange(1.0, 101.0), np.arange(1.0, 101.0)).to_csv(data, index=False)
    result = analyze_dataset(data, signal_cols=["price"], output_dir=tmp_path / "out")
    assert result["selected_signals"] == ["price"] and result["transforms_applied"] == {}
    assert "## Signal transforms" not in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
