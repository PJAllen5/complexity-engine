#!/usr/bin/env python3
"""Regenerate data/physio_example.csv, the synthetic heart-rate-like example dataset.

Each subject has one 5-minute recording per condition, sampled at 1 Hz (300 samples),
long enough for every time-series metric to be defined per recording (DFA needs at
least 16 samples; sample entropy needs matching templates). The two conditions differ
in dynamics, not just level, so the example report has something real to interpret:

- rest: lower mean, strongly persistent fluctuations (AR(1), phi=0.95) plus a visible
  respiratory-like oscillation (0.25 Hz)
- task: higher mean, weaker persistence (phi=0.6) and a damped oscillation

The data is synthetic and illustrates the pipeline only; it is not a physiological model.
Run from the repository root: python data/make_physio_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 300
SUBJECT_OFFSETS = {"s01": 0.0, "s02": 2.5, "s03": -1.5}
CONDITIONS = {
    "rest": {"start": "2026-09-21T00:00:00", "mean": 72.0, "phi": 0.95, "noise_sd": 0.6, "oscillation": 1.5},
    "task": {"start": "2026-09-21T00:10:00", "mean": 84.0, "phi": 0.6, "noise_sd": 1.0, "oscillation": 0.4},
}


def _recording(rng: np.random.Generator, mean: float, phi: float, noise_sd: float, oscillation: float) -> np.ndarray:
    fluctuation = np.zeros(N_SAMPLES)
    for i in range(1, N_SAMPLES):
        fluctuation[i] = phi * fluctuation[i - 1] + rng.normal(scale=noise_sd)
    t = np.arange(N_SAMPLES)
    return mean + fluctuation + oscillation * np.sin(2 * np.pi * 0.25 * t)


def main() -> None:
    rng = np.random.default_rng(20260921)
    frames = []
    for subject, offset in SUBJECT_OFFSETS.items():
        for condition, params in CONDITIONS.items():
            signal = _recording(rng, params["mean"] + offset, params["phi"], params["noise_sd"], params["oscillation"])
            frames.append(
                pd.DataFrame(
                    {
                        "subject_id": subject,
                        "condition": condition,
                        "timestamp": pd.date_range(params["start"], periods=N_SAMPLES, freq="s").strftime("%Y-%m-%dT%H:%M:%S"),
                        "signal": np.round(signal, 1),
                    }
                )
            )
    output_path = Path(__file__).resolve().parent / "physio_example.csv"
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
