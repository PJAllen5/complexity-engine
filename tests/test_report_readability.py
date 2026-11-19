"""Readable reports and plots: tables render as real HTML tables with sensible number formatting,
and plots show one point per recording instead of every sample."""

import numpy as np
import pandas as pd

from complexity_engine.analysis import _as_table, _format_cell, analyze_dataset
from complexity_engine.plots import plot_complexity_metrics, plot_condition_comparison


def test_cells_are_rounded_but_whole_numbers_stay_whole():
    assert _format_cell(72.254667) == "72.25"
    assert _format_cell(0.000123456) == "0.0001235"
    assert _format_cell(300.0) == "300" and _format_cell(np.int64(7)) == "7"
    assert _format_cell(float("nan")) == "—" and _format_cell(None) == "—"
    assert _format_cell(True) == "True" and _format_cell("a|b") == "a\\|b"


def test_tables_align_numbers_right_and_text_left():
    lines = _as_table(pd.DataFrame({"condition": ["rest"], "mean": [1.23456], "flag": [True]}))
    assert lines[0] == "| condition | mean | flag |"
    assert lines[1] == "| :--- | ---: | :--- |"
    assert lines[2] == "| rest | 1.235 | True |"
    assert _as_table(pd.DataFrame()) == ["(no rows)", ""]


def test_html_report_has_real_tables_and_no_text_dumps(tmp_path):
    result = analyze_dataset("data/physio_example.csv", output_dir=tmp_path)
    html = open(result["report_html"], encoding="utf-8").read()
    assert html.count("<table>") >= 4 and "<pre>" not in html
    assert "prefers-color-scheme: dark" in html and "background: var(--surface)" in html


def _frame(n_per_unit, units=3):
    rng = np.random.default_rng(0)
    rows = [{"condition": c, "subject_id": f"s{u}", "signal": rng.normal(10 if c == "a" else 12)}
            for c in ("a", "b") for u in range(units) for _ in range(n_per_unit)]
    return pd.DataFrame(rows)


def test_condition_plot_handles_units_many_samples_and_few_samples(tmp_path):
    for name, frame, units in [("units", _frame(500), ["subject_id"]), ("many", _frame(500), []),
                               ("few", _frame(5), [])]:
        path = tmp_path / f"{name}.png"
        assert plot_condition_comparison(frame, "signal", "condition", path, unit_cols=units) == str(path)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_metrics_plot_wraps_panels_and_draws_surrogates(tmp_path):
    metrics = ["sample_entropy", "spectral_entropy", "dfa_exponent", "dominant_period"]
    units = pd.DataFrame({"signal": "x", "condition": ["a", "a", "b", "b"],
                          **{m: [1.0, 1.1, 1.5, 1.6] for m in metrics},
                          "dfa_exponent_surrogate_mean": [0.5] * 4})
    summary = units.groupby(["condition", "signal"], as_index=False)[metrics].mean()
    path = tmp_path / "metrics.png"
    assert plot_complexity_metrics(summary, "x", metrics, path, unit_complexity=units) == str(path)
    from PIL import Image
    width, height = Image.open(path).size
    assert height > width / 3  # four metrics wrap onto two rows instead of one long strip
