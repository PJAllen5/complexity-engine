"""axis: time, duration scales in metric_parameters, and declared value units (PR 3 of
docs/design/time-axis-and-node-attributes.md)."""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset, resolve_duration_parameters
from complexity_engine.domain_profiles import DomainProfile, parse_duration
from complexity_engine.metrics import dfa_exponent


@pytest.mark.parametrize(
    "text, seconds",
    [("30D", 30 * 86_400), ("12h", 43_200), ("15min", 900), ("2W", 1_209_600), ("0.5s", 0.5), (" 3D ", 259_200)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("value", ["30", "30d", "30m", "D", "30 days", 30, None])
def test_parse_duration_rejects_non_durations(value):
    assert parse_duration(value) is None


def _mapping(axis="time", **dfa):
    return {"name": "t", "recommended_metrics": ["dfa_exponent"], "axis": axis,
            "metric_parameters": {"dfa_exponent": dfa}}


@pytest.mark.parametrize(
    "data, message",
    [
        (_mapping(axis="samples", min_scale="30D"), "axis is 'samples'"),
        (_mapping(min_scale="30 days"), "isn't a positive duration"),
        (_mapping(min_scale="0D"), "isn't a positive duration"),
        (_mapping(order="2D"), "only min_scale, max_scale may be given as text"),
        ({"name": "t", "axis": "clock"}, "axis must be one of"),
        ({"name": "t", "units": ["degC"]}, "units must map column names"),
        ({"name": "t", "units": {"x": ""}}, "units must map column names"),
    ],
)
def test_profile_rejects_invalid_time_settings(data, message):
    with pytest.raises(ValueError, match=message):
        DomainProfile.from_mapping(data)


def test_default_axis_is_samples_and_integers_still_count_samples():
    profile = DomainProfile.from_mapping({"name": "t", "metric_parameters": {"dfa_exponent": {"min_scale": 30}}})
    assert profile.axis == "samples" and not profile.has_duration_parameters()
    assert resolve_duration_parameters({"dfa_exponent": {"min_scale": 30, "max_scale": "2D"}}, 3_600) == {
        "dfa_exponent": {"min_scale": 30, "max_scale": 48}
    }


def _write_profile(tmp_path, name, axis="time", min_scale="'1D'", max_scale="'8D'", extra=""):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
{name}:
  name: {name}
  description: Test profile.
  recommended_metrics: [dfa_exponent]
  axis: {axis}
  metric_parameters:
    dfa_exponent:
      min_scale: {min_scale}
      max_scale: {max_scale}
{extra}
""",
        encoding="utf-8",
    )
    return path


def _series_csv(tmp_path, freq, periods, irregular_unit=None, time_values=None):
    """Two conditions x two units of seeded noise sampled every `freq`."""
    rng = np.random.default_rng(3)
    frames = []
    for unit, condition in (("s1", "rest"), ("s2", "rest"), ("s3", "task"), ("s4", "task")):
        times = pd.date_range("2001-01-01", periods=periods, freq=freq)
        if unit == irregular_unit:
            jitter = rng.uniform(-0.4, 0.4, periods) * pd.Timedelta(f"1{freq}")
            times = times + pd.to_timedelta(jitter)
        frames.append(pd.DataFrame({
            "subject_id": unit, "condition": condition,
            "timestamp": time_values if time_values is not None else times.strftime("%Y-%m-%d %H:%M:%S"),
            "signal": rng.standard_normal(periods),
        }))
    path = tmp_path / f"series_{freq}.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


def test_durations_become_samples_at_each_units_interval(tmp_path):
    csv_path = _series_csv(tmp_path, "h", 24 * 40)
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="hourly_timed",
        domain_config=_write_profile(tmp_path, "hourly_timed"), output_dir=tmp_path / "out",
    )
    s1 = pd.read_csv(csv_path).query("subject_id == 's1'")["signal"].to_numpy()
    units = result["complexity_by_unit"].set_index("subject_id")
    # 1D and 8D at one sample per hour are 24 and 192 samples.
    assert units.loc["s1", "dfa_exponent"] == pytest.approx(dfa_exponent(s1, min_scale=24, max_scale=192))
    assert (units["not_computed"] == "").all()
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "- Time axis: metric scales are in time (axis: time)" in report
    assert "min_scale=1D = 24 samples at 1h, max_scale=8D = 192 samples at 1h" in report


def test_whole_number_scales_still_count_samples_on_a_time_axis(tmp_path):
    csv_path = _series_csv(tmp_path, "h", 24 * 40)
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="hourly_samples",
        domain_config=_write_profile(tmp_path, "hourly_samples", min_scale=10, max_scale=100),
        output_dir=tmp_path / "out",
    )
    s1 = pd.read_csv(csv_path).query("subject_id == 's1'")["signal"].to_numpy()
    units = result["complexity_by_unit"].set_index("subject_id")
    assert units.loc["s1", "dfa_exponent"] == pytest.approx(dfa_exponent(s1, min_scale=10, max_scale=100))


def test_irregular_units_are_not_computed_on_a_time_axis(tmp_path):
    csv_path = _series_csv(tmp_path, "h", 24 * 40, irregular_unit="s2")
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="jittered",
        domain_config=_write_profile(tmp_path, "jittered"), output_dir=tmp_path / "out",
    )
    units = result["complexity_by_unit"].set_index("subject_id")
    assert np.isnan(units.loc["s2", "dfa_exponent"])
    assert units.loc["s2", "not_computed"].startswith("irregularly sampled")
    assert units.drop("s2")["dfa_exponent"].notna().all()
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "they aren't computed for these units" in report
    assert "No metrics for 1 recording(s) of signal" in report and "rest/s2: irregularly sampled" in report


def test_irregular_units_are_still_computed_on_a_samples_axis(tmp_path):
    csv_path = _series_csv(tmp_path, "h", 24 * 40, irregular_unit="s2")
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="jittered_samples",
        domain_config=_write_profile(tmp_path, "jittered_samples", axis="samples", min_scale=10, max_scale=100),
        output_dir=tmp_path / "out",
    )
    units = result["complexity_by_unit"]
    assert units["dfa_exponent"].notna().all() and "not_computed" not in units.columns


def test_durations_need_dates_not_plain_numbers(tmp_path):
    csv_path = _series_csv(tmp_path, "h", 200, time_values=np.arange(200))
    with pytest.raises(ValueError, match="holds plain numbers"):
        analyze_dataset(
            csv_path, signal_cols=["signal"], domain="numeric_time",
            domain_config=_write_profile(tmp_path, "numeric_time"), output_dir=tmp_path / "out",
        )


def test_time_axis_needs_a_time_column(tmp_path):
    df = pd.DataFrame({"condition": ["rest"] * 50 + ["task"] * 50, "signal": np.arange(100, dtype=float)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="sets axis: time, which needs a readable time column"):
        analyze_dataset(
            csv_path, signal_cols=["signal"], domain="no_time",
            domain_config=_write_profile(tmp_path, "no_time"), output_dir=tmp_path / "out",
        )


def test_declared_units_are_reported_on_the_signal_level_only(tmp_path):
    csv_path = _series_csv(tmp_path, "D", 400)
    extra = "  units:\n    signal: mmHg\n    pressure: kPa\n"
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="with_units",
        domain_config=_write_profile(tmp_path, "with_units", extra=extra), output_dir=tmp_path / "out",
    )
    assert set(result["condition_summary"]["unit"]) == {"mmHg"}
    significance = result["significance_summary"].set_index("measure")
    assert significance.loc["signal_mean", "unit"] == "mmHg"
    assert significance.loc["dfa_exponent", "unit"] == ""
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "- Units: signal in mmHg, pressure in kPa (not in this dataset)" in report


def test_profiles_without_units_keep_their_output_columns(tmp_path):
    csv_path = _series_csv(tmp_path, "D", 400)
    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="no_units",
        domain_config=_write_profile(tmp_path, "no_units", axis="samples", min_scale=8, max_scale=50),
        output_dir=tmp_path / "out",
    )
    assert "unit" not in result["condition_summary"].columns
    assert "unit" not in result["significance_summary"].columns
