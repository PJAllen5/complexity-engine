"""Download GHCN-Daily station records and convert them to a daily-temperature-anomaly CSV.

Eichner et al. (2003) ran DFA on deseasonalized daily maximum temperatures from 95 stations
and found long-range persistence with a fluctuation exponent alpha ~0.65 at continental
(inland and coastal) stations, but a higher one, averaging ~0.8, on islands. That published
contrast makes it a known-answer check for the time-series pipeline outside biology.

The stations are the ones Eichner et al. name in their Figs. 1-3, matched to GHCN-Daily
(Menne et al., 2012). Wellington (NZ) is left out because GHCN-Daily has no long record for
it, and the two summit stations (Zugspitze, Wushaoling) because the published contrast is
island vs continental. GHCN-Daily records don't always match the paper's: some cover
different years, and Plymouth's is a different site (Mountbatten, from 1959).

Inclusion rules, fixed before any exponent was computed:
- TMAX values that passed all GHCN quality checks (empty QFLAG).
- At least MIN_VALID_DAYS valid days, enough for DFA up to 3000-day windows.
- At least MIN_COVERAGE of the days between the first and last valid value present.
- Where two GHCN records exist for one station, the one with more valid days.

Each station is deseasonalized as in Eichner et al.: subtract the mean of that calendar date
(e.g. every 2 March) over all of the station's years. Missing days are then dropped rather
than filled, so the series runs over valid days only.

Station files (~17 MB in total) are cached under data/external/ghcnd/by_station/ and the
output is written to data/external/ghcn_tmax_anomaly.csv. data/external/ is not committed;
GHCN-Daily is distributed by NOAA NCEI without use restrictions -- cite Menne et al. (2012).

Output columns: station_id, station, condition (island/continental), setting
(island/coastal/inland), time (ISO date), tmax_anomaly (degrees C).

Usage:
    python data/fetch_ghcn_temperature.py
"""

from __future__ import annotations

import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/"

# (station as named by Eichner et al., setting, candidate GHCN-Daily IDs)
STATIONS = [
    ("Vienna", "inland", ["AU000005901"]),
    ("Perm", "inland", ["RSM00028224"]),
    ("Jena", "inland", ["GM000004204"]),
    ("Brno", "inland", ["EZM00011723"]),
    ("Chita", "inland", ["RSM00030758"]),
    ("Tashkent", "inland", ["UZM00038457"]),
    ("Potsdam", "inland", ["GM000003342"]),
    ("Minsk", "inland", ["BOM00026850"]),
    ("Oxford", "inland", ["UK000056225"]),
    ("Cheyenne", "inland", ["USW00024018"]),
    ("Kunming", "inland", ["CHM00056778"]),
    ("Charleston", "coastal", ["USW00013782"]),
    ("Pusan", "coastal", ["KSW00043213", "KSM00047159"]),
    ("Hong Kong", "coastal", ["CHM00045005"]),
    ("Anadyr", "coastal", ["RSM00025563"]),
    ("Hamburg", "coastal", ["GM000010147"]),
    ("Plymouth", "coastal", ["UKE00105870"]),
    ("Feodosiya", "coastal", ["UPM00033976"]),
    ("Thursday Island", "island", ["ASN00027022"]),
    ("Koror", "island", ["PSW00040309"]),
    ("Raoul Island", "island", ["NZ000093994"]),
    ("Campbell Island", "island", ["NZ000939450"]),
    ("Wrangel Island", "island", ["RSM00021982"]),
]

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "external" / "ghcnd" / "by_station"
OUTPUT_CSV = DATA_DIR / "external" / "ghcn_tmax_anomaly.csv"

MIN_VALID_DAYS = 12_000
MIN_COVERAGE = 0.8


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(station_id: str) -> Path:
    path = CACHE_DIR / f"{station_id}.csv.gz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(f"{BASE_URL}{station_id}.csv.gz", context=_ssl_context()) as response:
                path.write_bytes(response.read())
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLCertVerificationError):
                raise SystemExit(
                    f"Couldn't verify NOAA's TLS certificate ({error.reason}). This Python has no "
                    "CA certificates configured: `pip install certifi`, or on macOS run "
                    "'Install Certificates.command' from your Python folder. Alternatively, download "
                    f"the by_station .csv.gz files yourself into {CACHE_DIR} and rerun."
                ) from error
            raise
    return path


def read_tmax(path: Path) -> pd.Series:
    """Quality-controlled daily maximum temperature (degrees C), indexed by date."""
    raw = pd.read_csv(
        path,
        header=None,
        names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obs_time"],
        dtype={"date": str, "element": str, "qflag": str},
        usecols=["date", "element", "value", "qflag"],
    )
    tmax = raw[(raw["element"] == "TMAX") & raw["qflag"].isna() & (raw["value"] != -9999)]
    return pd.Series(tmax["value"].to_numpy() / 10.0, index=pd.to_datetime(tmax["date"], format="%Y%m%d")).sort_index()


def coverage(tmax: pd.Series) -> float:
    if tmax.empty:
        return 0.0
    span_days = (tmax.index[-1] - tmax.index[0]).days + 1
    return len(tmax) / span_days


def deseasonalize(tmax: pd.Series) -> pd.Series:
    """Subtract each calendar date's mean over all years (Eichner et al., 2003)."""
    calendar_date = tmax.index.strftime("%m-%d")
    return tmax - tmax.groupby(calendar_date).transform("mean")


def main() -> int:
    frames = []
    for station, setting, candidates in STATIONS:
        records = {station_id: read_tmax(_download(station_id)) for station_id in candidates}
        station_id = max(records, key=lambda key: len(records[key]))
        tmax = records[station_id]
        span = f"{tmax.index[0].year}-{tmax.index[-1].year}" if len(tmax) else "no data"
        status = f"{station} ({station_id}, {span}): {len(tmax)} valid days, {coverage(tmax):.0%} coverage"
        if len(tmax) < MIN_VALID_DAYS or coverage(tmax) < MIN_COVERAGE:
            print(f"{status} -- excluded (needs >= {MIN_VALID_DAYS} days and >= {MIN_COVERAGE:.0%} coverage)")
            continue
        print(f"{status} -- included")
        anomaly = deseasonalize(tmax)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "station": station,
                    "condition": "island" if setting == "island" else "continental",
                    "setting": setting,
                    "time": anomaly.index.strftime("%Y-%m-%d"),
                    "tmax_anomaly": anomaly.to_numpy(),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")
    counts = df.drop_duplicates("station_id")["condition"].value_counts().to_dict()
    print(f"Wrote {len(df)} daily anomalies from {sum(counts.values())} stations {counts} to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
