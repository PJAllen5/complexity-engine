import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset, sampling_summary


def _daily(dates, unit="s1", condition="rest"):
    return pd.DataFrame(
        {"subject_id": unit, "condition": condition, "date": dates, "signal": np.arange(len(dates), dtype=float)}
    )


def test_regular_daily_series_has_no_gaps():
    df = _daily(pd.date_range("2000-01-01", periods=10, freq="D").strftime("%Y-%m-%d"))
    row = sampling_summary(df, unit_cols=["subject_id"], time_col="date").iloc[0]
    assert (row["interval"], row["coverage"], row["n_gaps"], row["irregular"]) == ("1D", 1.0, 0, False)


def test_missing_days_are_reported_as_gaps():
    dates = pd.date_range("2000-01-01", periods=10, freq="D").delete([3, 4, 7])
    row = sampling_summary(_daily(dates), unit_cols=["subject_id"], time_col="date").iloc[0]
    assert row["interval"] == "1D"
    assert row["n_gaps"] == 2
    assert row["longest_gap"] == "3D"
    assert row["coverage"] == pytest.approx(7 / 10)
    assert not row["irregular"]


def test_irregular_numeric_spacing_is_flagged():
    times = np.cumsum(np.random.default_rng(0).uniform(0.5, 1.4, size=200))
    df = pd.DataFrame({"condition": "rest", "time": times, "signal": 1.0})
    row = sampling_summary(df, time_col="time").iloc[0]
    assert row["spacing_cv"] > 0.1
    assert row["irregular"]


def test_repeated_timestamps_are_counted_not_used_as_spacings():
    df = pd.DataFrame({"condition": "rest", "time": [0, 1, 1, 2, 3], "signal": 1.0})
    row = sampling_summary(df, time_col="time").iloc[0]
    assert (row["interval"], row["n_gaps"], row["duplicate_times"], row["coverage"]) == ("1", 0, 1, 1.0)


def test_each_unit_is_summarized_separately():
    df = pd.concat(
        [
            _daily(pd.date_range("2000-01-01", periods=5, freq="D"), unit="s1"),
            _daily(pd.date_range("2000-01-01", periods=5, freq="h"), unit="s2"),
        ]
    )
    rows = sampling_summary(df, unit_cols=["subject_id"], time_col="date").set_index("subject_id")
    assert rows.loc["s1", "interval"] == "1D"
    assert rows.loc["s2", "interval"] == "1h"


def test_no_time_column_gives_an_empty_summary():
    df = pd.DataFrame({"condition": "rest", "signal": [1.0, 2.0]})
    assert sampling_summary(df, time_col=None).empty


def test_report_includes_sampling_section_and_csv(tmp_path):
    dates = pd.date_range("2000-01-01", periods=60, freq="D")
    df = pd.concat(
        [_daily(dates.delete([10, 11]), unit="s1", condition="rest"), _daily(dates, unit="s2", condition="task")]
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(csv_path, signal_cols=["signal"], output_dir=tmp_path / "out")

    assert result["sampling_by_unit"].set_index("subject_id").loc["s1", "n_gaps"] == 1
    assert (tmp_path / "out" / "sampling_by_unit.csv").exists()
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Sampling" in report
    assert "1 gap(s) in 1 of 2 unit(s)" in report
    assert "sorted by date" in report  # 'date' is recognized as the time column


def test_report_says_when_there_is_no_time_column(tmp_path):
    df = pd.DataFrame({"condition": ["rest"] * 20 + ["task"] * 20, "signal": np.arange(40, dtype=float)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(csv_path, signal_cols=["signal"], output_dir=tmp_path / "out")

    assert result["sampling_csv"] is None
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "No time column" in report


# --- gap_policy --------------------------------------------------------------------------

from complexity_engine.domain_profiles import DomainProfile  # noqa: E402
from complexity_engine.metrics import Segments, dfa_exponent, sample_entropy, spectral_entropy  # noqa: E402


def test_a_single_segment_matches_the_plain_series_exactly():
    x = np.random.default_rng(1).standard_normal(3000)
    for metric in (sample_entropy, spectral_entropy, dfa_exponent):
        assert metric(Segments([x])) == metric(x), metric.__name__
    assert dfa_exponent(Segments([x]), min_scale=10, max_scale=300, order=2) == dfa_exponent(
        x, min_scale=10, max_scale=300, order=2
    )


def test_segmented_dfa_ignores_a_level_shift_across_a_gap():
    # Two white-noise records at different levels: joining them makes the step look like
    # long-range persistence; computing within segments sees white noise (~0.5).
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(4000), rng.standard_normal(4000) + 3.0
    assert dfa_exponent(np.concatenate([a, b]), min_scale=16, max_scale=512) > 0.7
    assert dfa_exponent(Segments([a, b]), min_scale=16, max_scale=512) == pytest.approx(0.5, abs=0.07)


def test_segmented_dfa_needs_four_windows_of_max_scale_across_segments():
    rng = np.random.default_rng(3)
    three = Segments([rng.standard_normal(150) for _ in range(3)])  # 3 windows of 100
    four = Segments([rng.standard_normal(150) for _ in range(4)])
    assert np.isnan(dfa_exponent(three, min_scale=10, max_scale=100))
    assert np.isfinite(dfa_exponent(four, min_scale=10, max_scale=100))


def _brute_force_segmented_sample_entropy(segments, m=2, r=0.2):
    tol = r * np.std(np.concatenate(segments))

    def matches(level):
        templates = [seg[i : i + level] for seg in segments for i in range(len(seg) - level + 1)]
        return sum(
            np.max(np.abs(templates[i] - templates[j])) <= tol
            for i in range(len(templates))
            for j in range(i + 1, len(templates))
        )

    return -np.log(matches(m + 1) / matches(m))


def test_segmented_sample_entropy_forms_templates_within_segments_only():
    rng = np.random.default_rng(4)
    segments = [rng.standard_normal(n) for n in (40, 3, 55)]
    expected = _brute_force_segmented_sample_entropy(segments)
    assert sample_entropy(Segments(segments)) == pytest.approx(expected, abs=1e-12)
    assert sample_entropy(Segments(segments)) != sample_entropy(np.concatenate(segments))


def test_segmented_spectral_entropy_is_a_length_weighted_mean_of_usable_segments():
    rng = np.random.default_rng(5)
    a, b, short = rng.standard_normal(200), rng.standard_normal(100), rng.standard_normal(10)
    expected = (200 * spectral_entropy(a) + 100 * spectral_entropy(b)) / 300
    assert spectral_entropy(Segments([a, short, b])) == pytest.approx(expected)


@pytest.mark.parametrize("data, message", [({"gap_policy": "skip"}, "gap_policy must be one of"),
                                           ({"min_coverage": 1.5}, "min_coverage must be in")])
def test_profile_rejects_invalid_gap_settings(data, message):
    with pytest.raises(ValueError, match=message):
        DomainProfile.from_mapping({"name": "t", **data})


def _write_profile(tmp_path, gap_policy, min_coverage=0.8):
    path = tmp_path / "profile.yaml"
    path.write_text(
        f"""
gappy:
  name: gappy
  description: Test profile.
  recommended_metrics: [dfa_exponent]
  gap_policy: {gap_policy}
  min_coverage: {min_coverage}
""",
        encoding="utf-8",
    )
    return path


def _gappy_csv(tmp_path):
    rng = np.random.default_rng(6)
    dates = pd.date_range("2000-01-01", periods=2000, freq="D")
    frames = []
    units = (("s1", "rest", slice(800, 1200)), ("s2", "task", slice(0, 0)), ("s3", "rest", slice(0, 0)),
             ("s4", "task", slice(0, 0)))
    for unit, condition, drop in units:
        frame = _daily(dates.strftime("%Y-%m-%d"), unit=unit, condition=condition)
        frame["signal"] = rng.standard_normal(len(frame))
        frames.append(frame.drop(frame.index[drop]))
    path = tmp_path / "gappy.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


def test_segment_policy_computes_within_segments_and_says_so(tmp_path):
    result = analyze_dataset(
        _gappy_csv(tmp_path), signal_cols=["signal"], domain="gappy",
        domain_config=_write_profile(tmp_path, "segment"), output_dir=tmp_path / "out",
    )
    units = result["complexity_by_unit"].set_index("subject_id")
    assert units.loc["s1", "n_segments"] == 2 and units.loc["s2", "n_segments"] == 1
    s1 = pd.read_csv(_gappy_csv(tmp_path)).query("subject_id == 's1'")["signal"].to_numpy()
    expected = dfa_exponent(Segments([s1[:800], s1[800:]]))
    assert units.loc["s1", "dfa_exponent"] == pytest.approx(expected)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "- Gap policy: segment" in report
    assert "no DFA window or entropy template spans a gap" in report


def test_exclude_policy_drops_low_coverage_units_and_lists_them(tmp_path):
    result = analyze_dataset(
        _gappy_csv(tmp_path), signal_cols=["signal"], domain="gappy",
        domain_config=_write_profile(tmp_path, "exclude", 0.9), output_dir=tmp_path / "out",
    )
    assert "s1" not in set(result["complexity_by_unit"]["subject_id"])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Excluded for coverage below 90%: s1 (80%)" in report


def test_gap_policy_other_than_join_needs_a_time_column(tmp_path):
    df = pd.DataFrame({"condition": ["rest"] * 50 + ["task"] * 50, "signal": np.arange(100, dtype=float)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="needs a readable time column"):
        analyze_dataset(
            csv_path, signal_cols=["signal"], domain="gappy",
            domain_config=_write_profile(tmp_path, "segment"), output_dir=tmp_path / "out",
        )


def test_max_joined_gap_joins_short_gaps_and_splits_long_ones():
    from complexity_engine.analysis import _gap_starts

    days = np.concatenate([np.arange(0, 100), np.arange(101, 200), np.arange(260, 300)])  # 1 and 60 days missing
    assert list(_gap_starts(days.astype(float))) == [100, 199]
    assert list(_gap_starts(days.astype(float), max_joined_gap=1)) == [199]
    assert list(_gap_starts(days.astype(float), max_joined_gap=59)) == [199]
    assert list(_gap_starts(days.astype(float), max_joined_gap=60)) == []


def test_profile_rejects_invalid_max_joined_gap():
    for value in (-1, 2.5, True):
        with pytest.raises(ValueError, match="max_joined_gap must be"):
            DomainProfile.from_mapping({"name": "t", "max_joined_gap": value})
