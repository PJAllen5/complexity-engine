"""The installed commands (complexity_engine.cli) work the same as the run_*.py scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from complexity_engine import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_timeseries_command_runs_in_process(tmp_path):
    cli.timeseries_main(["--data", str(REPO_ROOT / "data" / "physio_example.csv"), "--output-dir", str(tmp_path)])
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "complexity_summary.csv").exists()


def test_network_command_runs_in_process(tmp_path):
    cli.network_main(["--data", str(REPO_ROOT / "data" / "network_example.csv"), "--output-dir", str(tmp_path)])
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "network_metrics_summary.csv").exists()


def test_checkout_keeps_example_data_default():
    # Run from the repository, --data falls back to the bundled example (installed copies require it).
    assert cli._IN_CHECKOUT
    assert cli._example("physio_example.csv") == str(REPO_ROOT / "data" / "physio_example.csv")


def test_unknown_domain_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown domain"):
        cli.timeseries_main(["--domain", "not_a_domain", "--output-dir", str(tmp_path)])


def test_pyproject_packages_the_domain_profiles():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"complexity_engine.config" = "config"' in text
    assert "complexity-analysis" in text and "complexity-network" in text
