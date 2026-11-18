"""Write occipital EEG from the PhysioNet EEG Motor Movement/Imagery Database as a CSV.

EEGMMIDB (Schalk et al., 2004; Goldberger et al., 2000): 109 subjects, 64-channel EEG at 160 Hz
recorded with BCI2000. Runs 1 and 2 are one-minute resting baselines with eyes open and eyes
closed. Only the Oz channel (central occipital, where alpha is strongest) of those two runs is
kept. Open Data Commons Attribution v1.0.

The EDF files are read here, at fetch time (roadmap item E7), so the engine itself stays
CSV-only: EDF stores each channel as 16-bit integers with a linear scale to physical units (uV)
given in its header. Each EDF is deleted after conversion, so only the CSV is kept.

Output, under data/external/ (not committed):
- eeg_eyes.csv: subject_id, condition (eyes_open, eyes_closed), sampling_rate_hz, time (s), oz_uv

Usage:
    python data/fetch_eeg_eyes.py
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://physionet.org/files/eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf"
RUNS = {1: "eyes_open", 2: "eyes_closed"}
CHANNEL = "Oz"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = EXTERNAL_DIR / "eegmmidb"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def read_edf_channel(raw: bytes, label: str) -> tuple[np.ndarray, float]:
    """One channel of an EDF/EDF+ file as physical values, and its sampling rate in Hz."""
    def field(start: int, width: int, count: int) -> list[str]:
        return [raw[start + i * width : start + (i + 1) * width].decode("ascii").strip() for i in range(count)]

    n_records = int(raw[236:244])
    record_seconds = float(raw[244:252])
    ns = int(raw[252:256])
    offset = 256
    labels = field(offset, 16, ns); offset += 16 * ns
    offset += 80 * ns + 8 * ns  # transducer, physical dimension
    phys_min = np.array(field(offset, 8, ns), dtype=float); offset += 8 * ns
    phys_max = np.array(field(offset, 8, ns), dtype=float); offset += 8 * ns
    dig_min = np.array(field(offset, 8, ns), dtype=float); offset += 8 * ns
    dig_max = np.array(field(offset, 8, ns), dtype=float); offset += 8 * ns
    offset += 80 * ns  # prefiltering
    samples = np.array(field(offset, 8, ns), dtype=int); offset += 8 * ns
    offset += 32 * ns  # reserved
    names = [name.rstrip(".").lower() for name in labels]
    index = names.index(label.lower())
    record_size = int(samples.sum())
    data = np.frombuffer(raw[offset : offset + 2 * record_size * n_records], dtype="<i2").reshape(n_records, record_size)
    start = int(samples[:index].sum())
    digital = data[:, start : start + samples[index]].ravel().astype(float)
    gain = (phys_max[index] - phys_min[index]) / (dig_max[index] - dig_min[index])
    return (digital - dig_min[index]) * gain + phys_min[index], samples[index] / record_seconds


def _download(subject: int, run: int) -> bytes:
    cached = CACHE_DIR / f"S{subject:03d}R{run:02d}.edf"
    if cached.exists():
        return cached.read_bytes()
    url = URL.format(subject=subject, run=run)
    try:
        with urllib.request.urlopen(url, context=_ssl_context()) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise SystemExit(f"Couldn't download {url} ({error.reason}). Download it yourself to {cached} and rerun.") from error


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for subject in range(1, 110):
        for run, condition in RUNS.items():
            values, rate = read_edf_channel(_download(subject, run), CHANNEL)
            frames.append(pd.DataFrame({
                "subject_id": f"S{subject:03d}", "condition": condition, "sampling_rate_hz": rate,
                "time": np.round(np.arange(values.size) / rate, 5), "oz_uv": np.round(values, 3),
            }))
            cached = CACHE_DIR / f"S{subject:03d}R{run:02d}.edf"
            if cached.exists():
                cached.unlink()
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(EXTERNAL_DIR / "eeg_eyes.csv", index=False)
    rates = data.groupby(["subject_id", "condition"])["sampling_rate_hz"].first().value_counts().to_dict()
    print(f"Wrote eeg_eyes.csv: {data['subject_id'].nunique()} subjects, recordings by sampling rate {rates}")


if __name__ == "__main__":
    main()
