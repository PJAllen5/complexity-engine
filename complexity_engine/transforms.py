"""Signal transforms a domain profile can apply before metrics (roadmap item E1).

A transform makes a signal from a column through a list of steps, applied in order within each
recording, in time order:

    transforms:
      returns: {from: price, steps: [log_return]}
      volatility: {from: price, steps: [log_return, abs]}
      rr_interval: [zscore]          # shorthand: transform the column in place

Steps: diff, log_return, abs, zscore, bandpass(low_hz, high_hz) and envelope. bandpass needs the
sampling rate, from a date/time column or the profile's sampling_interval.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Step name -> number of numeric arguments.
TRANSFORM_STEPS = {"diff": 0, "log_return": 0, "abs": 0, "zscore": 0, "bandpass": 2, "envelope": 0}
# Steps that make the series one sample shorter (the first value has no predecessor).
_LOSES_FIRST_SAMPLE = {"diff", "log_return"}
_STEP_PATTERN = re.compile(r"^\s*([a-z_]+)\s*(?:\(([^()]*)\))?\s*$")


def parse_step(text: Any) -> tuple[str, tuple[float, ...]]:
    """'bandpass(8, 13)' -> ('bandpass', (8.0, 13.0)); raises ValueError for anything else."""
    match = _STEP_PATTERN.match(str(text))
    if match is None or match.group(1) not in TRANSFORM_STEPS:
        raise ValueError(f"unknown transform step '{text}'; steps are {', '.join(TRANSFORM_STEPS)}")
    name, raw_args = match.group(1), match.group(2)
    try:
        args = tuple(float(a) for a in raw_args.split(",")) if raw_args and raw_args.strip() else ()
    except ValueError:
        raise ValueError(f"transform step '{text}' has a non-numeric argument") from None
    if len(args) != TRANSFORM_STEPS[name]:
        raise ValueError(f"transform step '{name}' takes {TRANSFORM_STEPS[name]} argument(s), got {len(args)}")
    if name == "bandpass" and not 0 < args[0] < args[1]:
        raise ValueError(f"bandpass needs 0 < low < high, got {args}")
    return name, args


def parse_transforms(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate a profile's transforms mapping into {signal: {"from": column, "steps": [...]}},
    where steps are (name, args) pairs. Raises ValueError with what's wrong."""
    if not isinstance(raw, dict):
        raise ValueError("transforms must map a signal name to {from: column, steps: [...]} or a list of steps")
    parsed = {}
    for signal, spec in raw.items():
        spec = {"from": signal, "steps": spec} if isinstance(spec, list) else spec
        if not isinstance(spec, dict) or set(spec) - {"from", "steps"} or not isinstance(spec.get("steps"), list):
            raise ValueError(f"transforms.{signal} must be a list of steps or {{from: column, steps: [...]}}")
        if not spec["steps"]:
            raise ValueError(f"transforms.{signal} has no steps")
        try:
            steps = [parse_step(step) for step in spec["steps"]]
        except ValueError as error:
            raise ValueError(f"transforms.{signal}: {error}") from None
        parsed[str(signal)] = {"from": str(spec.get("from", signal)), "steps": steps}
    return parsed


def needs_sampling_rate(steps) -> bool:
    return any(name == "bandpass" for name, _ in steps)


def describe(steps) -> str:
    return ", ".join(f"{name}({', '.join(f'{a:g}' for a in args)})" if args else name for name, args in steps)


def apply_steps(values: np.ndarray, steps, sampling_rate: float | None = None) -> np.ndarray:
    """Apply steps to one recording's values (in time order). The result has the same length:
    steps that lose the first sample (diff, log_return) leave NaN there."""
    x = np.asarray(values, dtype=float)
    lost = 0
    for name, args in steps:
        body = x[lost:]
        if name in _LOSES_FIRST_SAMPLE:
            if name == "log_return":
                if np.any(body <= 0):
                    raise ValueError("log_return needs strictly positive values (e.g. prices)")
                body = np.log(body)
            body = np.diff(body)
            lost += 1
        elif name == "abs":
            body = np.abs(body)
        elif name == "zscore":
            sd = body.std()
            body = (body - body.mean()) / sd if sd > 0 else np.full_like(body, np.nan)
        elif name == "bandpass":
            body = _bandpass(body, *args, sampling_rate=sampling_rate)
        elif name == "envelope":
            from scipy.signal import hilbert

            body = np.abs(hilbert(body - body.mean()))
        x = np.concatenate([np.full(lost, np.nan), body])
    return x


def _bandpass(x: np.ndarray, low: float, high: float, sampling_rate: float | None) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass between low and high Hz."""
    from scipy.signal import butter, sosfiltfilt

    if not sampling_rate:
        raise ValueError("bandpass needs a sampling rate: a date/time column, or the profile's sampling_interval")
    if high >= sampling_rate / 2:
        raise ValueError(f"bandpass high edge {high:g} Hz must be below half the sampling rate ({sampling_rate / 2:g} Hz)")
    sos = butter(4, [low, high], btype="band", fs=sampling_rate, output="sos")
    if x.size <= 3 * (2 * sos.shape[0] + 1):
        return np.full_like(x, np.nan)  # too short to filter; the rows are dropped and counted
    return sosfiltfilt(sos, x)
