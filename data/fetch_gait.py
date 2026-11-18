"""Write stride-interval series from PhysioNet's gait databases as CSVs.

- Gait in Aging and Disease (gaitdb): 5 healthy young (21-29 years) and 5 healthy elderly
  (71-77 years) adults, 15 minutes of walking around a circular path, plus 5 people with
  Parkinson's disease. One stride interval per row.
- Gait in Neurodegenerative Disease (gaitndd): 20 Huntington's disease, 15 Parkinson's,
  13 ALS and 16 controls, about 5 minutes of walking. The left foot's stride interval is used.
  subject-description.txt gives each subject's age and, for Huntington's disease, total
  functional capacity (TFC: higher = less impaired).

These are the groups of Hausdorff et al. (1997), from the same lab, though the subjects
available on PhysioNet may not be exactly the paper's. Both databases are open
(Open Data Commons Attribution v1.0; cite Goldberger et al., 2000 and the papers above).

Preparation, fixed before any exponent was computed: for each subject, strides more than
3 SD from that subject's mean stride interval (turns, stumbles, missed foot strikes) are
removed, as in Hausdorff's analyses. The files already start after the first ~20 s of walking.

Output, under data/external/ (not committed):
- gait_aging.csv: subject_id, condition (young, elderly, parkinson), age, stride, stride_interval
- gait_ndd.csv: subject_id, condition (control, huntington, parkinson, als), age, severity,
  stride, stride_interval

Usage:
    python data/fetch_gait.py
"""

from __future__ import annotations

import io
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://physionet.org/files/{db}/1.0.0/{name}"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = EXTERNAL_DIR / "gait"

GAITDB_RECORDS = [f"y{i}-{age}-si.txt" for i, age in enumerate((23, 29, 23, 21, 26), 1)] + [
    f"o{i}-{age}-si.txt" for i, age in enumerate((76, 74, 75, 77, 71), 1)
] + [f"pd{i}-si.txt" for i in range(1, 6)]
GAITNDD_GROUPS = {"control": 16, "hunt": 20, "park": 15, "als": 13}
GAITNDD_CONDITIONS = {"control": "control", "hunt": "huntington", "park": "parkinson", "als": "als"}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _fetch(db: str, name: str) -> str:
    cached = CACHE_DIR / db / name
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = BASE.format(db=db, name=name)
        try:
            with urllib.request.urlopen(url, context=_ssl_context()) as response:
                cached.write_bytes(response.read())
        except urllib.error.URLError as error:
            raise SystemExit(
                f"Couldn't download {url} ({error.reason}). Download it yourself to {cached} and rerun."
            ) from error
    return cached.read_text(encoding="utf-8")


def remove_outlier_strides(intervals: pd.Series, n_sd: float = 3.0) -> pd.Series:
    """Strides within n_sd standard deviations of the subject's mean stride interval."""
    mean, sd = intervals.mean(), intervals.std()
    return intervals[(intervals - mean).abs() <= n_sd * sd]


def _subject_frame(subject_id: str, condition: str, intervals: pd.Series, **metadata) -> tuple[pd.DataFrame, int]:
    kept = remove_outlier_strides(intervals.dropna().reset_index(drop=True))
    frame = pd.DataFrame({
        "subject_id": subject_id,
        "condition": condition,
        **metadata,
        "stride": range(1, len(kept) + 1),
        "stride_interval": kept.to_numpy(),
    })
    return frame, int(intervals.notna().sum() - len(kept))


def build_aging() -> pd.DataFrame:
    frames, removed = [], 0
    for name in GAITDB_RECORDS:
        table = pd.read_csv(io.StringIO(_fetch("gaitdb", name)), sep=r"\s+", header=None, names=["time", "interval"])
        subject = name.removesuffix("-si.txt")
        prefix = re.match(r"[a-z]+", subject).group(0)
        condition = {"y": "young", "o": "elderly", "pd": "parkinson"}[prefix]
        age = int(subject.split("-")[1]) if "-" in subject else pd.NA
        frame, n_removed = _subject_frame(subject, condition, table["interval"], age=age)
        frames.append(frame)
        removed += n_removed
    print(f"gaitdb: removed {removed} outlier stride(s) across {len(frames)} subjects")
    return pd.concat(frames, ignore_index=True)


def build_ndd() -> pd.DataFrame:
    description = pd.read_csv(io.StringIO(_fetch("gaitndd", "subject-description.txt")), sep="\t")
    description = description.rename(columns={description.columns[0]: "subject"}).set_index("subject")
    frames, removed = [], 0
    for prefix, count in GAITNDD_GROUPS.items():
        for i in range(1, count + 1):
            subject = f"{prefix}{i}"
            table = pd.read_csv(io.StringIO(_fetch("gaitndd", f"{subject}.ts")), sep=r"\s+", header=None)
            row = description.loc[subject] if subject in description.index else None
            metadata = {
                "age": pd.to_numeric(row["AGE(YRS)"], errors="coerce") if row is not None else pd.NA,
                "severity": pd.to_numeric(row["Duration/Severity"], errors="coerce") if row is not None else pd.NA,
            }
            frame, n_removed = _subject_frame(subject, GAITNDD_CONDITIONS[prefix], table[1], **metadata)
            frames.append(frame)
            removed += n_removed
    print(f"gaitndd: removed {removed} outlier stride(s) across {len(frames)} subjects")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    aging, ndd = build_aging(), build_ndd()
    aging.to_csv(EXTERNAL_DIR / "gait_aging.csv", index=False)
    ndd.to_csv(EXTERNAL_DIR / "gait_ndd.csv", index=False)
    print(f"Wrote gait_aging.csv ({len(aging)} strides) and gait_ndd.csv ({len(ndd)} strides)")


if __name__ == "__main__":
    main()
