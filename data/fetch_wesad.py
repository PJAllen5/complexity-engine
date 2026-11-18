"""Derive heart rate and skin conductance from WESAD for the baseline and stress conditions.

WESAD (Schmidt et al., 2018): 15 subjects wearing a chest RespiBAN (700 Hz) through a baseline
period and the Trier Social Stress Test (TSST; Kirschbaum et al., 1993), among other conditions.
Licensed for academic, non-commercial use only; see the dataset's page. Download WESAD.zip
(about 2.25 GB) from https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ yourself, accepting
its terms, to data/external/wesad/WESAD.zip. This script reads only the plain-text files in the
archive (SX_respiban.txt, SX_quest.csv), streaming one subject at a time, and never loads the
SX.pkl pickle files: unpickling a downloaded file can run arbitrary code.

Per subject, prepared at fetch time (roadmap item E7):
- Baseline ("Base") and stress ("TSST") segments are cut from the RespiBAN recording using the
  protocol times in SX_quest.csv, given as minutes.seconds (7.08 is 7 min 8 s).
- ECG and EDA are converted to mV and uS with the formulas in WESAD's readme (vcc = 3, 16 bits).
- R-peaks are detected in the ECG (5-15 Hz band-pass, squared slope, peaks at least 0.33 s apart
  above a third of the segment's 99th percentile). RR intervals outside 0.3-2.0 s, or more than
  20% from the median of the 11 around them, are dropped as detection errors or ectopic beats.
- EDA is averaged into 0.25 s bins (4 Hz).

Output, under data/external/ (not committed):
- wesad_heart.csv: subject_id, condition (baseline, stress), time (s, of each beat), rr_interval (s), heart_rate (bpm)
- wesad_eda.csv: subject_id, condition, time (s), eda_us

Usage:
    python data/fetch_wesad.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RATE = 700
VCC, CHAN_BIT = 3.0, 2**16
CONDITIONS = {"Base": "baseline", "TSST": "stress"}

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
ARCHIVE = EXTERNAL_DIR / "wesad" / "WESAD.zip"


def minutes_seconds(value: str) -> float:
    """'7.08' (7 min 8 s) -> 428 seconds. A single digit after the point is a dropped trailing
    zero ('50.3' is 50 min 30 s): the files write seconds under 10 with a leading zero ('26.04')."""
    minutes, _, seconds = str(value).strip().partition(".")
    return 60 * int(minutes) + (int(seconds.ljust(2, "0")) if seconds else 0)


def protocol_times(quest: str) -> dict[str, tuple[float, float]]:
    rows = {line.split(";")[0].strip("# ").upper(): line.split(";")[1:] for line in quest.splitlines() if line.startswith("#")}
    order = [c for c in rows["ORDER"] if c]
    return {name: (minutes_seconds(rows["START"][i]), minutes_seconds(rows["END"][i])) for i, name in enumerate(order)}


def r_peaks(ecg: np.ndarray, rate: int = RATE) -> np.ndarray:
    from scipy.signal import butter, find_peaks, sosfiltfilt

    filtered = sosfiltfilt(butter(2, [5, 15], btype="band", fs=rate, output="sos"), ecg)
    energy = np.diff(filtered, prepend=filtered[0]) ** 2
    peaks, _ = find_peaks(energy, distance=int(0.33 * rate), height=np.percentile(energy, 99) / 3)
    return peaks


def clean_rr(beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, RR intervals) with implausible and ectopic intervals removed."""
    rr = np.diff(beat_times)
    times = beat_times[1:]
    plausible = (rr >= 0.3) & (rr <= 2.0)
    local = pd.Series(rr).rolling(11, center=True, min_periods=1).median().to_numpy()
    keep = plausible & (np.abs(rr - local) <= 0.2 * local)
    return times[keep], rr[keep]


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"WESAD isn't downloaded: save WESAD.zip to {ARCHIVE} (see this script's docstring).")
    heart, eda, dropped = [], [], []
    with zipfile.ZipFile(ARCHIVE) as archive:
        subjects = sorted({n.split("/")[1] for n in archive.namelist() if n.endswith("_respiban.txt")},
                          key=lambda s: int(s[1:]))
        for subject in subjects:
            times = protocol_times(archive.read(f"WESAD/{subject}/{subject}_quest.csv").decode("utf-8"))
            with archive.open(f"WESAD/{subject}/{subject}_respiban.txt") as handle:
                raw = pd.read_csv(io.TextIOWrapper(handle), sep="\t", comment="#", header=None,
                                  usecols=[2, 3], names=["ecg", "eda"], dtype=float)
            for name, condition in CONDITIONS.items():
                start, end = (int(t * RATE) for t in times[name])
                segment = raw.iloc[start:end]
                ecg = (segment["ecg"].to_numpy() / CHAN_BIT - 0.5) * VCC
                beats = (start + r_peaks(ecg)) / RATE
                beat_times, rr = clean_rr(beats)
                dropped.append(1 - rr.size / max(beats.size - 1, 1))
                heart.append(pd.DataFrame({"subject_id": subject, "condition": condition, "time": np.round(beat_times, 4),
                                           "rr_interval": np.round(rr, 4), "heart_rate": np.round(60 / rr, 2)}))
                conductance = (segment["eda"].to_numpy() / CHAN_BIT) * VCC / 0.12
                bins = conductance[: conductance.size // (RATE // 4) * (RATE // 4)].reshape(-1, RATE // 4).mean(axis=1)
                eda.append(pd.DataFrame({"subject_id": subject, "condition": condition,
                                         "time": np.round(start / RATE + np.arange(bins.size) * 0.25, 3),
                                         "eda_us": np.round(bins, 4)}))
            print(f"{subject}: baseline {times['Base']}, stress {times['TSST']} (s)")
    pd.concat(heart).to_csv(EXTERNAL_DIR / "wesad_heart.csv", index=False)
    pd.concat(eda).to_csv(EXTERNAL_DIR / "wesad_eda.csv", index=False)
    print(f"Wrote wesad_heart.csv and wesad_eda.csv for {len(subjects)} subjects; "
          f"median share of RR intervals dropped {np.median(dropped):.1%}")


if __name__ == "__main__":
    main()
