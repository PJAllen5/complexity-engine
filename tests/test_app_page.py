"""The app's time-series page (docs/design/app.md, PR 2): the helpers behind it, and the page itself
driven through Streamlit's test harness (skipped if Streamlit isn't installed)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.app.choices import (
    combined_preview,
    custom_profiles,
    domain_choices,
    example_upload,
    read_upload,
    timeseries_defaults,
)
from complexity_engine.app.runner import Upload

PAGE = Path(__file__).resolve().parents[1] / "complexity_engine" / "app" / "streamlit_app.py"


def _csv(frame: pd.DataFrame, name="data.csv") -> Upload:
    return Upload(name, frame.to_csv(index=False).encode())


def test_domain_choices_split_by_pipeline():
    timeseries = {c.name for c in domain_choices("timeseries")}
    network = {c.name for c in domain_choices("network")}
    assert {"physiology", "gait", "eeg_oscillations"} <= timeseries
    assert {"network", "social_network"} <= network and not timeseries & network


def test_an_uploaded_profile_is_offered_and_validated():
    good = Upload("mine.yaml", b"mine:\n  name: mine\n  description: Mine.\n  recommended_metrics: [dfa_exponent]\n")
    assert "mine" in {c.name for c in domain_choices("timeseries", good)}
    bad = Upload("bad.yaml", b"bad:\n  name: bad\n  axis: clock\n")
    with pytest.raises(ValueError, match="axis must be one of"):
        custom_profiles(bad)


def test_preview_merges_files_with_the_same_columns():
    a, b = pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [3]})
    assert len(combined_preview([_csv(a, "a.csv"), _csv(b, "b.csv")])) == 3
    with pytest.raises(ValueError, match="different columns"):
        combined_preview([_csv(a, "a.csv"), _csv(pd.DataFrame({"y": [1]}), "b.csv")])


def test_excel_and_unsupported_files():
    frame = pd.DataFrame({"condition": ["a", "b"], "signal": [1.0, 2.0]})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    # Excel stores 1.0 as 1, so compare values, not dtypes.
    pd.testing.assert_frame_equal(read_upload(Upload("d.xlsx", buffer.getvalue())), frame, check_dtype=False)
    with pytest.raises(ValueError, match="unsupported file type"):
        read_upload(Upload("d.txt", b"x"))


def test_defaults_are_the_engines_own_choices():
    defaults = timeseries_defaults(read_upload(example_upload("timeseries")))
    assert (defaults.signals, defaults.condition, defaults.units, defaults.time) == (
        ["signal"], "condition", ["subject_id"], "timestamp")


# --- the page itself ---------------------------------------------------------------------------

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def page():
    app = streamlit_testing.AppTest.from_file(str(PAGE), default_timeout=120)
    app.run()
    assert not app.exception
    return app


def test_example_run_matches_the_plain_cli_command(page):
    page.button(key="example_button").click().run()
    page.button(key="run").click().run()
    assert not page.exception and not page.error
    assert len(page.session_state["runs"]) == 1
    # Untouched choices pass "auto-detect" to the engine, so the command is the plain CLI call.
    assert page.code[0].value == ("complexity-analysis --data physio_example.csv --condition-col condition "
                                  "--domain physiology --output-dir results")
    assert page.success[0].value.startswith("Finished in")


def test_choices_reach_the_run_and_runs_accumulate(page):
    page.button(key="example_button").click().run()
    page.selectbox(key="domain").select("stress").run()
    page.radio(key="unit_mode").set_value("None: each condition is one recording").run()
    page.button(key="run").click().run()
    page.button(key="run").click().run()
    assert not page.exception and not page.error
    assert len(page.session_state["runs"]) == 2
    command = page.code[0].value
    assert "--domain stress" in command and "--unit-cols none" in command
    assert {s.key for s in page.selectbox} >= {"compare_a", "compare_b"}


def test_no_signal_columns_stops_before_running(page):
    page.button(key="example_button").click().run()
    page.multiselect(key="signals").set_value([]).run()
    assert "Choose at least one signal column." in [w.value for w in page.warning]
    assert not [b for b in page.button if b.key == "run"]


def test_network_page_points_to_the_cli_for_now(page):
    page.radio(key="analysis_type").set_value("Network").run()
    assert "complexity-network" in page.info[0].value


def test_launcher_runs_the_page(monkeypatch):
    import sys

    from streamlit.web import cli as streamlit_cli

    from complexity_engine.app import launch

    seen = {}
    monkeypatch.setattr(streamlit_cli, "main", lambda: seen.setdefault("argv", list(sys.argv)) and 0)
    monkeypatch.setattr(sys, "argv", ["complexity-app", "--server.port", "8600"])
    with pytest.raises(SystemExit):
        launch.main()
    assert seen["argv"] == ["streamlit", "run", str(PAGE), "--server.port", "8600"]
