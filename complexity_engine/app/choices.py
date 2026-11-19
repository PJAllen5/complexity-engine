"""What the page offers the user: file previews, domain choices and default columns.

No Streamlit here, so these are tested like any other module. Defaults come from the engine's own
auto-detection, so leaving every choice alone gives the same run as the CLI with no options.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis import _METRIC_FUNCTIONS, resolve_signal_columns, resolve_time_column, resolve_unit_columns
from ..domain_profiles import DomainProfile, load_domain_profiles
from ..network_metrics import NETWORK_METRIC_KEYS
from .runner import Upload

EXAMPLES = {
    "timeseries": Path(__file__).resolve().parents[2] / "data" / "physio_example.csv",
    "network": Path(__file__).resolve().parents[2] / "data" / "network_example.csv",
}
DATA_TYPES = ["csv", "xlsx", "xls"]


def example_upload(pipeline: str) -> Upload | None:
    """The bundled example file for a pipeline, or None in an installed copy without data/."""
    path = EXAMPLES[pipeline]
    return Upload(path.name, path.read_bytes()) if path.exists() else None


def read_upload(upload: Upload) -> pd.DataFrame:
    suffix = Path(upload.safe_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(upload.content))
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(io.BytesIO(upload.content))
        except ImportError as error:  # pandas needs openpyxl (.xlsx) or xlrd (.xls) to read Excel
            raise ValueError(f"{upload.safe_name}: reading Excel needs an extra package ({error}). "
                             "Save the sheet as CSV, or pip install openpyxl.") from None
    raise ValueError(f"{upload.safe_name}: unsupported file type {suffix or '(none)'}; upload CSV or Excel.")


def combined_preview(uploads: list[Upload]) -> pd.DataFrame:
    """All uploads read and stacked, as the engine merges several files. Files must share columns."""
    frames = [read_upload(u) for u in uploads]
    columns = list(frames[0].columns)
    for upload, frame in zip(uploads[1:], frames[1:]):
        if list(frame.columns) != columns:
            raise ValueError(f"{upload.safe_name} has different columns from {uploads[0].safe_name}; "
                             "files are merged, so they must share the same columns.")
    return pd.concat(frames, ignore_index=True)


@dataclass
class DomainChoice:
    name: str
    description: str
    question: str


def _is_timeseries(profile: DomainProfile) -> bool:
    return bool(profile.recommended_metrics) and set(profile.recommended_metrics) <= set(_METRIC_FUNCTIONS)


def _is_network(profile: DomainProfile) -> bool:
    return bool(profile.recommended_metrics) and set(profile.recommended_metrics) <= set(NETWORK_METRIC_KEYS)


def custom_profiles(upload: Upload) -> dict[str, DomainProfile]:
    """Profiles in an uploaded config file, validated as the engine validates them."""
    import json

    import yaml

    text = upload.content.decode("utf-8")
    payload = json.loads(text) if upload.safe_name.lower().endswith(".json") else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{upload.safe_name}: a profile file maps profile names to profiles.")
    return {DomainProfile.from_mapping(v, key=k).name: DomainProfile.from_mapping(v, key=k) for k, v in payload.items()}


def domain_choices(pipeline: str, custom: Upload | None = None) -> list[DomainChoice]:
    """Profiles that fit the pipeline, built-in first, then any from an uploaded config file."""
    fits = _is_timeseries if pipeline == "timeseries" else _is_network
    profiles = dict(load_domain_profiles())
    if custom is not None:
        profiles.update(custom_profiles(custom))
    return [DomainChoice(p.name, p.description, p.analysis_question)
            for name, p in profiles.items() if fits(p)]


@dataclass
class ColumnDefaults:
    signals: list[str]
    condition: str | None
    units: list[str]
    time: str | None


def timeseries_defaults(df: pd.DataFrame) -> ColumnDefaults:
    """What the engine would pick with no options: the defaults the page starts from."""
    try:
        signals = resolve_signal_columns(df)
    except ValueError:
        signals = []
    condition = "condition" if "condition" in df.columns else None
    units = resolve_unit_columns(df, None, condition_col=condition or "condition")
    return ColumnDefaults(signals=signals, condition=condition, units=units, time=resolve_time_column(df))
