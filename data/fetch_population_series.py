"""Write population time series with published cycles as CSVs.

- Canadian lynx trappings, 1821-1934 (114 years): annual numbers of lynx trapped in the
  Mackenzie River district, from Hudson's Bay Company records (Elton & Nicholson, 1942). This
  copy is R's `datasets::lynx` (Brockwell & Davis, 1991, Series G; the series analyzed by
  Campbell & Walker, 1977), downloaded from the Rdatasets archive.

Output, under data/external/ (not committed; cite the papers above when you use the data):
- lynx.csv: series_id, condition, time (year), count

Usage:
    python data/fetch_population_series.py
"""

from __future__ import annotations

import io
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

LYNX_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE = EXTERNAL_DIR / "population" / "lynx_rdatasets.csv"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(url: str, cached: Path) -> str:
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, context=_ssl_context()) as response:
                cached.write_bytes(response.read())
        except urllib.error.URLError as error:
            raise SystemExit(f"Couldn't download {url} ({error.reason}). Download it yourself to {cached} and rerun.") from error
    return cached.read_text(encoding="utf-8")


def write_lynx() -> None:
    raw = pd.read_csv(io.StringIO(_download(LYNX_URL, CACHE)))
    lynx = pd.DataFrame({"series_id": "mackenzie_lynx", "condition": "lynx", "time": raw["time"].astype(int),
                         "count": raw["value"].astype(float)})
    lynx.to_csv(EXTERNAL_DIR / "lynx.csv", index=False)
    print(f"Wrote lynx.csv: {len(lynx)} years, {lynx['time'].min()}-{lynx['time'].max()}")


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    write_lynx()


if __name__ == "__main__":
    main()
