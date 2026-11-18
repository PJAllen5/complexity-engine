"""Write long eyes-closed resting EEG from OpenNeuro ds004504 as CSVs.

ds004504 (Miltiadous et al., 2023): routine clinical EEG, 19 channels at 500 Hz, eyes closed at
rest, from 29 healthy older adults (12.5-16.5 min each), 36 people with Alzheimer's disease and 23
with frontotemporal dementia. CC0. Only the healthy (C) and Alzheimer's (A) groups are used.

Prepared here, at fetch time (roadmap item E7): the unprocessed recordings (sub-XXX, not the
dataset's ASR-cleaned derivatives, which cut out stretches and so would hide gaps) are read from
EEGLAB .set files with scipy.io.loadmat, which reads data and never runs code. Seven channels are
kept: O1 and O2 (occipital) and T5, P3, Pz, P4, T6 (temporo-parietal). Each is decimated from 500 Hz
to 125 Hz with a zero-phase anti-aliasing filter, which preserves everything below about 50 Hz,
including the alpha band. Each .set file is deleted after conversion.

Output, under data/external/ (not committed; cite Miltiadous et al., 2023):
- resting_eeg_occipital.csv: subject_id, condition (healthy, alzheimers), time (s), o1, o2 (uV)
- resting_eeg_temporoparietal.csv: subject_id, condition, time (s), t5, p3, pz, p4, t6 (uV)

Usage:
    python data/fetch_resting_eeg.py
"""

from __future__ import annotations

import io
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://s3.amazonaws.com/openneuro.org/ds004504/"
GROUPS = {"C": "healthy", "A": "alzheimers"}
OCCIPITAL = ("O1", "O2")
TEMPOROPARIETAL = ("T5", "P3", "Pz", "P4", "T6")
TARGET_RATE = 125

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = EXTERNAL_DIR / "ds004504"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _fetch(name: str, cached: Path) -> Path:
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(BASE + name, context=_ssl_context()) as response:
                cached.write_bytes(response.read())
        except urllib.error.URLError as error:
            raise SystemExit(f"Couldn't download {BASE + name} ({error.reason}). Download it yourself to {cached} and rerun.") from error
    return cached


def read_set(path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Channels (uV) and sampling rate from an EEGLAB .set file with the data stored inside it."""
    from scipy.io import loadmat

    contents = loadmat(path, squeeze_me=True, struct_as_record=False)
    # EEGLAB saves fields either inside an EEG struct or at the top level of the file.
    get = (lambda key: getattr(contents["EEG"], key)) if "EEG" in contents else contents.__getitem__
    rate = float(get("srate"))
    labels = [str(ch.labels) for ch in np.atleast_1d(get("chanlocs"))]
    data = np.asarray(get("data"), dtype=float)
    if data.shape[0] != len(labels):
        data = data.T
    return dict(zip(labels, data)), rate


def main() -> None:
    from scipy.signal import decimate

    participants = pd.read_csv(io.StringIO(_fetch("participants.tsv", CACHE_DIR / "participants.tsv").read_text()), sep="\t")
    occipital, temporoparietal = [], []
    for _, row in participants[participants["Group"].isin(GROUPS)].iterrows():
        subject = row["participant_id"]
        name = f"{subject}/eeg/{subject}_task-eyesclosed_eeg.set"
        path = _fetch(name, CACHE_DIR / f"{subject}_task-eyesclosed_eeg.set")
        channels, rate = read_set(path)
        factor = int(round(rate / TARGET_RATE))
        if abs(rate / factor - TARGET_RATE) > 1e-6:
            raise SystemExit(f"{subject}: sampling rate {rate} Hz isn't a multiple of {TARGET_RATE} Hz")
        kept = {ch: decimate(channels[ch], factor, ftype="fir", zero_phase=True) for ch in (*OCCIPITAL, *TEMPOROPARIETAL)}
        n = len(kept["O1"])
        base = {"subject_id": subject, "condition": GROUPS[row["Group"]], "time": np.round(np.arange(n) / TARGET_RATE, 4)}
        occipital.append(pd.DataFrame({**base, **{ch.lower(): np.round(kept[ch], 2) for ch in OCCIPITAL}}))
        temporoparietal.append(pd.DataFrame({**base, **{ch.lower(): np.round(kept[ch], 2) for ch in TEMPOROPARIETAL}}))
        path.unlink()
    pd.concat(occipital).to_csv(EXTERNAL_DIR / "resting_eeg_occipital.csv", index=False)
    pd.concat(temporoparietal).to_csv(EXTERNAL_DIR / "resting_eeg_temporoparietal.csv", index=False)
    counts = participants[participants["Group"].isin(GROUPS)]["Group"].map(GROUPS).value_counts().to_dict()
    print(f"Wrote resting_eeg_occipital.csv and resting_eeg_temporoparietal.csv: {counts}")


if __name__ == "__main__":
    main()
