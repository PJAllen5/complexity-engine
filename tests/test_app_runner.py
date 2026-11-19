"""The app's runner and the settings it shares with the CLI (docs/design/app.md, PR 1).

The design's "done" criteria checked here:
1. Parity: every CLI option is a settings field, and settings round-trip through the CLI.
2. Same results: the runner writes CSVs byte-identical to the CLI's, on the bundled examples
   (always) and on validation datasets CI downloads (skipped if absent).
3. Downloads: the zip holds the report, CSVs, plots and settings.json, and settings.json's command
   repeats the run exactly.
Also: nothing is left on disk after a run, and engine errors reach the caller unchanged.
"""

from __future__ import annotations

import dataclasses
import io
import json
import shlex
import tomllib
import zipfile
from pathlib import Path

import pytest

import complexity_engine
from complexity_engine.app.runner import Upload, run_analysis
from complexity_engine.cli import build_network_parser, build_timeseries_parser, network_main, timeseries_main
from complexity_engine.settings import NetworkSettings, TimeSeriesSettings

ROOT = Path(__file__).resolve().parents[1]
DATA, EXTERNAL = ROOT / "data", ROOT / "data" / "external"
MAINS = {"timeseries": timeseries_main, "network": network_main}


def _dests(parser):
    return {action.dest for action in parser._actions if action.dest != "help"}


def _fields(settings_type):
    return {f.name for f in dataclasses.fields(settings_type)}


@pytest.mark.parametrize("parser, settings_type", [(build_timeseries_parser(), TimeSeriesSettings),
                                                   (build_network_parser(), NetworkSettings)])
def test_every_cli_option_is_a_settings_field(parser, settings_type):
    assert _dests(parser) == _fields(settings_type)


@pytest.mark.parametrize("settings", [
    TimeSeriesSettings(data=["a.csv", "b.csv"], signal_cols=["x", "y"], unit_cols=[], domain="gait",
                       domain_config="p.yaml", output_dir="out"),
    TimeSeriesSettings(data=["a.csv"]),
    NetworkSettings(data=["e.csv"], weight_col=None, condition_col=None, directed=True, compute_centrality=True,
                    compute_small_world=True, compute_communities=True, nodes="n.csv", node_col="id",
                    attributes=["team", "age"], permutations=99, path_sources=7, random_seed=3, domain="social_network"),
    NetworkSettings(data=["e.csv"]),
])
def test_settings_round_trip_through_the_command_line(settings):
    parser = build_timeseries_parser() if isinstance(settings, TimeSeriesSettings) else build_network_parser()
    assert type(settings).from_cli(parser.parse_args(settings.cli_args())) == settings


def _zip_csvs(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return {Path(n).name: archive.read(n) for n in archive.namelist() if n.endswith(".csv")}


def _dir_csvs(folder):
    return {p.name: p.read_bytes() for p in Path(folder).glob("*.csv")}


def _upload(path):
    return Upload(Path(path).name, Path(path).read_bytes())


CASES = [
    ("timeseries", TimeSeriesSettings(data=[str(DATA / "physio_example.csv")]), None),
    ("network", NetworkSettings(data=[str(DATA / "network_example.csv")], compute_centrality=True,
                                compute_communities=True), None),
    ("timeseries", TimeSeriesSettings(data=[str(EXTERNAL / "gait_aging.csv")], signal_cols=["stride_interval"],
                                      unit_cols=["subject_id"], domain="gait"), None),
    ("timeseries", TimeSeriesSettings(data=[str(EXTERNAL / "lynx.csv")], signal_cols=["count"],
                                      unit_cols=["series_id"], domain="population_dynamics"), None),
    ("network", NetworkSettings(data=[str(EXTERNAL / "karate_club_edges.csv")], weight_col=None, condition_col=None,
                                compute_communities=True, attributes=["faction"], domain="social_network"),
     EXTERNAL / "karate_club_factions.csv"),
]


@pytest.mark.parametrize("pipeline, settings, nodes", CASES, ids=lambda c: getattr(c, "data", [""])[0].split("/")[-1] if hasattr(c, "data") else None)
def test_runner_matches_the_cli_exactly(tmp_path, pipeline, settings, nodes):
    if not all(Path(p).exists() for p in settings.data) or (nodes and not nodes.exists()):
        pytest.skip("validation data not downloaded")
    cli_settings = dataclasses.replace(settings, output_dir=str(tmp_path / "cli"),
                                       **({"nodes": str(nodes)} if nodes else {}))
    MAINS[pipeline](cli_settings.cli_args())
    result = run_analysis(pipeline, [_upload(p) for p in settings.data], settings,
                          node_table=_upload(nodes) if nodes else None)
    cli_csvs = _dir_csvs(tmp_path / "cli")
    assert cli_csvs and _zip_csvs(result.zip_bytes) == cli_csvs


def test_zip_contents_and_settings_json(tmp_path):
    settings = TimeSeriesSettings(data=[str(DATA / "physio_example.csv")])
    result = run_analysis("timeseries", [_upload(DATA / "physio_example.csv")], settings)
    with zipfile.ZipFile(io.BytesIO(result.zip_bytes)) as archive:
        names = set(archive.namelist())
    assert {"results/report.html", "results/report.md", "results/settings.json",
            "results/complexity_by_unit.csv"} <= names
    assert any(n.startswith("results/plots/") and n.endswith(".png") for n in names)
    recorded = json.loads(result.settings_json)
    assert recorded["analysis"] == "timeseries" and recorded["engine_version"] == complexity_engine.__version__
    assert recorded["settings"]["data"] == ["physio_example.csv"]
    assert recorded["command"].startswith("complexity-analysis --data physio_example.csv")
    assert "<html" in result.report_html.lower() and "complexity_by_unit.csv" in result.tables
    assert not Path(result.workdir).exists()


def test_settings_json_command_repeats_the_run(tmp_path, monkeypatch):
    edges, nodes = EXTERNAL / "karate_club_edges.csv", EXTERNAL / "karate_club_factions.csv"
    if not (edges.exists() and nodes.exists()):
        pytest.skip("validation data not downloaded")
    settings = NetworkSettings(data=[str(edges)], weight_col=None, condition_col=None, attributes=["faction"],
                               permutations=99, domain="social_network")
    result = run_analysis("network", [_upload(edges)], settings, node_table=_upload(nodes))
    for path in (edges, nodes):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    monkeypatch.chdir(tmp_path)
    program, *argv = shlex.split(json.loads(result.settings_json)["command"])
    assert program == "complexity-network"
    network_main(argv)
    assert _dir_csvs(tmp_path / "results") == _zip_csvs(result.zip_bytes)


def test_upload_names_cannot_escape_the_run_folder():
    assert Upload("../../etc/data.csv", b"").safe_name == "data.csv"
    assert Upload("C:\\Users\\me\\data.csv", b"").safe_name == "data.csv"
    with pytest.raises(ValueError, match="usable file name"):
        Upload("..", b"").safe_name


def test_duplicate_upload_names_are_rejected():
    content = (DATA / "physio_example.csv").read_bytes()
    with pytest.raises(ValueError, match="both named"):
        run_analysis("timeseries", [Upload("a/x.csv", content), Upload("b/x.csv", content)],
                     TimeSeriesSettings(data=["x.csv"]))


def test_engine_errors_reach_the_caller_and_leave_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(ValueError, match="Unknown domain"):
        run_analysis("timeseries", [_upload(DATA / "physio_example.csv")],
                     TimeSeriesSettings(data=["x"], domain="no_such_domain"))
    assert list(tmp_path.iterdir()) == []


def test_bad_calls_are_rejected():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        run_analysis("images", [Upload("a.csv", b"")], TimeSeriesSettings(data=["a.csv"]))
    with pytest.raises(TypeError, match="NetworkSettings"):
        run_analysis("network", [Upload("a.csv", b"")], TimeSeriesSettings(data=["a.csv"]))
    with pytest.raises(ValueError, match="at least one data file"):
        run_analysis("timeseries", [], TimeSeriesSettings(data=[]))


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert complexity_engine.__version__ == project["version"]
