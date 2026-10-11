"""Download the PhysioNet Fantasia database and convert it to an RR-interval CSV.

Fantasia (Iyengar et al., 1996; Goldberger et al., 2000) holds 120-minute supine resting
ECG recordings of 20 young (21-34 yr) and 20 elderly (68-85 yr) healthy subjects, with
annotated heartbeats. Its published results -- elderly hearts lose the long-range fractal
correlations of young hearts -- make it a known-answer check for the time-series pipeline.

Only the beat-annotation (.ecg) and header (.hea) files are downloaded (~0.7 MB in total);
the raw signals are not needed. Files are cached under data/external/fantasia/ and the
output CSV is written to data/external/fantasia_rr.csv. data/external/ is not committed:
the data is redistributed under the Open Data Commons Attribution License, so fetch it
from the source and cite Iyengar et al. (1996) and PhysioNet when you use it.

Output columns: subject_id, condition (young/elderly), age, sex, time (s, beat onset),
rr_interval (s). Only normal-to-normal (NN) intervals are kept -- both bounding beats
annotated normal (N) and 0.3 s <= RR <= 2.0 s -- the standard heart-rate-variability
preprocessing, since an ectopic beat or a missed detection produces an interval that
reflects the artifact rather than autonomic regulation. Exclusion counts are printed.

Usage:
    python data/fetch_fantasia.py
"""

from __future__ import annotations

import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://physionet.org/files/fantasia/1.0.0/"
RECORDS = [f"f{series}{group}{i:02d}" for series in (1, 2) for group in ("y", "o") for i in range(1, 11)]
CONDITION = {"y": "young", "o": "elderly"}

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "external" / "fantasia"
OUTPUT_CSV = DATA_DIR / "external" / "fantasia_rr.csv"

MIN_RR_SECONDS = 0.3
MAX_RR_SECONDS = 2.0

# MIT annotation-format codes (see PhysioNet's WFDB "annot" format documentation).
_NORMAL = 1
_SKIP, _NUM, _SUB, _CHN, _AUX = 59, 60, 61, 62, 63
# Codes that mark a heartbeat (WFDB's isqrs table); the rest are rhythm, noise, or comment
# markers, which must be dropped before differencing or they'd split a real interval in two.
_BEAT_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 30, 34, 35, 38, 41}


def read_mit_annotations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an MIT-format annotation file; return (sample_index, annotation_code) arrays.

    Each annotation is a little-endian 16-bit word: the top 6 bits are the code and the
    low 10 bits the sample-time increment. SKIP carries a longer increment in the next
    two words; NUM/SUB/CHN/AUX are modifiers of the previous annotation, not beats.
    """
    raw = path.read_bytes()
    words = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<u2")
    samples, codes = [], []
    time = 0
    i = 0
    while i < len(words):
        code, value = int(words[i]) >> 10, int(words[i]) & 0x3FF
        i += 1
        if code == 0 and value == 0:
            break
        if code == _SKIP:
            # 32-bit increment stored as high word then low word (PDP-11 order).
            time += np.int32((int(words[i]) << 16) | int(words[i + 1])).item()
            i += 2
        elif code == _AUX:
            i += (value + 1) // 2
        elif code in (_NUM, _SUB, _CHN):
            pass
        else:
            time += value
            samples.append(time)
            codes.append(code)
    return np.asarray(samples, dtype=np.int64), np.asarray(codes, dtype=np.int64)


def read_header(path: Path) -> dict:
    text = path.read_text()
    sampling_rate = float(text.split("\n", 1)[0].split()[2])
    age = re.search(r"Age:\s*(\d+)", text)
    sex = re.search(r"Sex:\s*(\w)", text)
    return {
        "sampling_rate": sampling_rate,
        "age": int(age.group(1)) if age else None,
        "sex": sex.group(1) if sex else None,
    }


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(record: str, suffix: str) -> Path:
    path = CACHE_DIR / f"{record}.{suffix}"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}{record}.{suffix}"
        try:
            with urllib.request.urlopen(url, context=_ssl_context()) as response:
                path.write_bytes(response.read())
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLCertVerificationError):
                raise SystemExit(
                    f"Couldn't verify PhysioNet's TLS certificate ({error.reason}). This Python has no "
                    "CA certificates configured: `pip install certifi`, or on macOS run "
                    "'Install Certificates.command' from your Python folder. Alternatively, download "
                    f"the .ecg and .hea files yourself into {CACHE_DIR} and rerun."
                ) from error
            raise
    return path


def nn_intervals(samples: np.ndarray, codes: np.ndarray, sampling_rate: float) -> tuple[pd.DataFrame, dict]:
    """Return NN intervals (beat time and RR, in seconds) and exclusion counts."""
    is_beat = np.isin(codes, list(_BEAT_CODES))
    samples, codes = samples[is_beat], codes[is_beat]
    beat_times = samples / sampling_rate
    rr = np.diff(beat_times)
    both_normal = (codes[:-1] == _NORMAL) & (codes[1:] == _NORMAL)
    in_range = (rr >= MIN_RR_SECONDS) & (rr <= MAX_RR_SECONDS)
    keep = both_normal & in_range
    counts = {
        "intervals": int(rr.size),
        "non_normal_beat": int((~both_normal).sum()),
        "out_of_range": int((both_normal & ~in_range).sum()),
    }
    return pd.DataFrame({"time": beat_times[1:][keep], "rr_interval": rr[keep]}), counts


def main() -> int:
    frames = []
    for record in RECORDS:
        header = read_header(_download(record, "hea"))
        samples, codes = read_mit_annotations(_download(record, "ecg"))
        intervals, counts = nn_intervals(samples, codes, header["sampling_rate"])
        print(
            f"{record}: {counts['intervals']} intervals, kept {len(intervals)} "
            f"(excluded {counts['non_normal_beat']} next to a non-normal beat, "
            f"{counts['out_of_range']} outside {MIN_RR_SECONDS}-{MAX_RR_SECONDS} s)"
        )
        frames.append(
            intervals.assign(
                subject_id=record,
                condition=CONDITION[record[2]],
                age=header["age"],
                sex=header["sex"],
            )[["subject_id", "condition", "age", "sex", "time", "rr_interval"]]
        )
    df = pd.concat(frames, ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
    print(f"Wrote {len(df)} NN intervals from {len(RECORDS)} recordings to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
