"""Run an analysis from uploaded files, for the app (docs/design/app.md).

No Streamlit here, so it can be tested like any other module. run_analysis() writes the uploads
to a fresh temporary folder, calls the same analysis function with the same arguments the CLI
would, reads every output into memory, builds a zip (report, CSVs, plots and settings.json), and
deletes the folder before returning. Nothing is left on disk after a run.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import __version__
from ..analysis import analyze_dataset
from ..network_analysis import analyze_network
from ..settings import NetworkSettings, TimeSeriesSettings, as_dict, command_line

PIPELINES = {"timeseries": (TimeSeriesSettings, analyze_dataset), "network": (NetworkSettings, analyze_network)}


@dataclass
class Upload:
    """A file the user provided: its name as uploaded, and its contents."""

    name: str
    content: bytes

    @property
    def safe_name(self) -> str:
        """The base name only, so a name like '../../x.csv' can't write outside the run folder."""
        name = Path(self.name.replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise ValueError(f"Not a usable file name: {self.name!r}")
        return name


@dataclass
class RunResult:
    """Everything a run produced, held in memory."""

    pipeline: str
    settings: dict
    command: str
    seconds: float
    report_html: str
    report_md: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: dict[str, bytes] = field(default_factory=dict)
    zip_bytes: bytes = b""
    settings_json: str = ""
    workdir: str = ""  # the temporary folder used, already deleted; kept so tests can check that


def _read_table(path: Path) -> pd.DataFrame:
    """A result CSV as a table. The engine writes some as empty files (e.g. cross-signal metrics
    when only one signal was analyzed), which become an empty table."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write(folder: Path, upload: Upload | None) -> str | None:
    if upload is None:
        return None
    path = folder / upload.safe_name
    if path.exists():
        raise ValueError(f"Two uploaded files are both named {upload.safe_name!r}; rename one.")
    path.write_bytes(upload.content)
    return str(path)


def run_analysis(
    pipeline: str,
    uploads: list[Upload],
    settings: TimeSeriesSettings | NetworkSettings,
    domain_config: Upload | None = None,
    node_table: Upload | None = None,
) -> RunResult:
    """Run one analysis on uploaded files and return its results in memory.

    `settings` is the shared settings object; its data, domain_config, nodes and output_dir are
    replaced by the uploads and the temporary folder. The engine's own errors (ValueError) are
    raised unchanged, so the app can show their message.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown analysis type {pipeline!r}; use one of {', '.join(PIPELINES)}.")
    settings_type, analyze = PIPELINES[pipeline]
    if not isinstance(settings, settings_type):
        raise TypeError(f"{pipeline} needs {settings_type.__name__}, got {type(settings).__name__}.")
    if not uploads:
        raise ValueError("Upload at least one data file.")

    # What settings.json records: the names the user knows, not temporary paths.
    recorded = replace(
        settings,
        data=[u.safe_name for u in uploads],
        domain_config=domain_config.safe_name if domain_config else None,
        output_dir="results",
        **({"nodes": node_table.safe_name if node_table else None} if pipeline == "network" else {}),
    )

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="complexity-run-") as workdir:
        folder = Path(workdir)
        inputs = folder / "input"
        inputs.mkdir()
        output = folder / "output"
        run_settings = replace(
            settings,
            data=[_write(inputs, u) for u in uploads],
            domain_config=_write(inputs, domain_config),
            output_dir=str(output),
            **({"nodes": _write(inputs, node_table)} if pipeline == "network" else {}),
        )
        analyze(**run_settings.analysis_kwargs())
        seconds = time.perf_counter() - start

        settings_json = json.dumps(
            {
                "analysis": pipeline,
                "engine_version": __version__,
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(seconds, 2),
                "settings": as_dict(recorded),
                "command": command_line(recorded),
                "note": "Run the command from a folder containing the files listed in settings.data "
                        "(and domain_config / nodes, if set) to repeat this analysis.",
            },
            indent=2,
        )
        (output / "settings.json").write_text(settings_json, encoding="utf-8")

        tables = {p.name: _read_table(p) for p in sorted(output.glob("*.csv"))}
        plots = {p.name: p.read_bytes() for p in sorted((output / "plots").glob("*.png"))}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("results") / path.relative_to(output))
        result = RunResult(
            pipeline=pipeline,
            settings=as_dict(recorded),
            command=command_line(recorded),
            seconds=seconds,
            report_html=(output / "report.html").read_text(encoding="utf-8"),
            report_md=(output / "report.md").read_text(encoding="utf-8"),
            tables=tables,
            plots=plots,
            zip_bytes=buffer.getvalue(),
            settings_json=settings_json,
            workdir=workdir,
        )
    return result
