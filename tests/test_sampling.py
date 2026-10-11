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
