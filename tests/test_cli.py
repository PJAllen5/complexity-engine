"""Subprocess-based smoke tests for the CLI entry points.

Every other test calls analyze_dataset()/analyze_network() directly, so nothing
actually exercised argparse itself -- a bad default, a broken flag, or a typo in
an option name would only be caught by running the CLI for real. These tests
don't try to cover every flag combination; they just confirm each entry point
actually runs end-to-end and produces its documented output files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_run_analysis_cli_with_default_data(tmp_path):
    result = _run_cli("run_analysis.py", ["--output-dir", str(tmp_path)])

    assert result.returncode == 0, result.stderr
    assert "Saved report to:" in result.stdout
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "condition_summary.csv").exists()
    assert (tmp_path / "complexity_summary.csv").exists()
    assert (tmp_path / "significance_summary.csv").exists()
    assert (tmp_path / "cross_signal_summary.csv").exists()
    assert any((tmp_path / "plots").glob("*.png"))


def test_run_analysis_cli_with_explicit_flags(tmp_path):
    result = _run_cli(
        "run_analysis.py",
        [
            "--data", "data/physio_example.csv",
            "--signal-cols", "signal",
            "--condition-col", "condition",
            "--domain", "attention",
            "--domain-config", "config/domain_profiles.yaml",
            "--output-dir", str(tmp_path),
        ],
    )

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Domain: attention" in report


def test_run_analysis_cli_merges_comma_separated_data_files(tmp_path):
    df1 = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 10.0]})
    df2 = pd.DataFrame({"condition": ["rest", "task"], "signal": [2.0, 11.0]})
    path1 = tmp_path / "session1.csv"
    path2 = tmp_path / "session2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)
    output_dir = tmp_path / "outputs"

    result = _run_cli(
        "run_analysis.py",
        ["--data", f"{path1},{path2}", "--signal-cols", "signal", "--output-dir", str(output_dir)],
    )

    assert result.returncode == 0, result.stderr
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Data source: session1.csv, session2.csv" in report
    assert "Input rows: 4" in report


def test_run_analysis_cli_rejects_unknown_domain(tmp_path):
    result = _run_cli("run_analysis.py", ["--domain", "not_a_real_domain", "--output-dir", str(tmp_path)])

    assert result.returncode != 0
    assert "Unknown domain" in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_run_network_analysis_cli_with_default_data(tmp_path):
    result = _run_cli("run_network_analysis.py", ["--output-dir", str(tmp_path)])

    assert result.returncode == 0, result.stderr
    assert "Saved report to:" in result.stdout
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "network_metrics_summary.csv").exists()
    assert (tmp_path / "plots" / "degree_distribution.png").exists()


def test_run_network_analysis_cli_directed_and_centrality_flags(tmp_path):
    result = _run_cli(
        "run_network_analysis.py",
        ["--directed", "--compute-centrality", "--output-dir", str(tmp_path)],
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "node_centrality.csv").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Directed: True" in report
    assert "## Node centrality" in report


def test_run_network_analysis_cli_rejects_unknown_domain(tmp_path):
    result = _run_cli("run_network_analysis.py", ["--domain", "not_a_real_domain", "--output-dir", str(tmp_path)])

    assert result.returncode != 0
    assert "Unknown domain" in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_run_network_analysis_cli_compute_small_world_flag(tmp_path):
    result = _run_cli(
        "run_network_analysis.py",
        ["--compute-small-world", "--random-seed", "7", "--output-dir", str(tmp_path)],
    )

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "### Small-world statistics" in report


def test_run_network_analysis_cli_merges_comma_separated_data_files(tmp_path):
    df1 = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "condition": ["rest", "rest"]})
    df2 = pd.DataFrame({"source": ["c", "a"], "target": ["a", "d"], "condition": ["task", "task"]})
    path1 = tmp_path / "session1.csv"
    path2 = tmp_path / "session2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)
    output_dir = tmp_path / "outputs"

    result = _run_cli("run_network_analysis.py", ["--data", f"{path1},{path2}", "--output-dir", str(output_dir)])

    assert result.returncode == 0, result.stderr
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Data source: session1.csv, session2.csv" in report
    assert "Input rows: 4" in report
