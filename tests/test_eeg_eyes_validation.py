"""Known-answer validation of the cognitive profile on resting EEG, roadmap item A5
(docs/design/validation-and-new-domains.md).

Known answer: the Berger effect. Closing the eyes boosts occipital alpha (8-13 Hz), which
concentrates spectral power and so lowers spectral entropy over the occipital cortex.
Data: PhysioNet EEGMMIDB (Schalk et al., 2004), runs 1 (eyes open) and 2 (eyes closed), one
minute each, channel Oz, 109 subjects.

Criteria, fixed before any metric was computed on these recordings:
1. Recordings not sampled at 160 Hz are excluded and counted.
2. With the cognitive profile (units: subject), Oz spectral entropy is lower with eyes closed
   in at least 70% of subjects with both recordings, and the per-recording Welch test of
   eyes_closed vs eyes_open gives p < 0.05 with the closed mean lower.
3. With eyes closed, dominant_period (linear detrending) is in the alpha band, 12.3-20 samples
   (8-13 Hz at 160 Hz), in more than half of subjects.

The data isn't committed (see data/fetch_eeg_eyes.py), so this file is skipped unless
data/external/eeg_eyes.csv exists.

Result: 1 passes; 2 and 3 fail (strict xfails). The raw recordings are dominated by slow drift
(median 45% of power below 1 Hz with eyes open, 31% closed, likely eye movements), which
concentrates the eyes-open spectrum and is the "dominant rhythm" for most subjects. The Berger
effect itself is plainly in the data (alpha share 0.24 closed vs 0.07 open; higher with eyes
closed in 92% of subjects), but that comparison was chosen after the failure. See
docs/validation/eeg_eyes.md, which also fixes the criteria for a band-passed rerun.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.metrics import dominant_period

EEG = Path(__file__).resolve().parents[1] / "data" / "external" / "eeg_eyes.csv"

pytestmark = pytest.mark.skipif(not EEG.exists(), reason="EEG not downloaded; run python data/fetch_eeg_eyes.py")


@pytest.fixture(scope="module")
def eeg(tmp_path_factory):
    data = pd.read_csv(EEG)
    kept = data[data["sampling_rate_hz"] == 160]
    folder = tmp_path_factory.mktemp("eeg")
    kept.to_csv(folder / "eeg_160hz.csv", index=False)
    result = analyze_dataset(folder / "eeg_160hz.csv", signal_cols=["oz_uv"], unit_cols=["subject_id"],
                             domain="cognitive", output_dir=folder / "out")
    excluded = data.groupby(["subject_id", "condition"])["sampling_rate_hz"].first().ne(160).sum()
    return result, kept, int(excluded)


def test_only_160_hz_recordings_are_analyzed(eeg):
    result, kept, excluded = eeg
    assert set(kept["sampling_rate_hz"]) == {160}
    assert len(result["complexity_by_unit"]) + excluded == 218


@pytest.mark.xfail(
    strict=True,
    reason="Raw EEG: slow drift below 1 Hz (45% of power with eyes open, 31% closed) concentrates the "
    "eyes-open spectrum, so spectral entropy is lower with eyes closed in only 42% of subjects "
    "(means 0.651 closed vs 0.635 open).",
)
def test_spectral_entropy_drops_when_the_eyes_close(eeg):
    result, _, _ = eeg
    units = result["complexity_by_unit"].pivot(index="subject_id", columns="condition", values="spectral_entropy")
    paired = units.dropna()
    assert (paired["eyes_closed"] < paired["eyes_open"]).mean() >= 0.7
    rows = result["significance_summary"]
    row = rows[(rows["measure"] == "spectral_entropy")].iloc[0]
    assert (row["condition_a"], row["condition_b"]) == ("eyes_closed", "eyes_open")
    assert row["mean_a"] < row["mean_b"] and row["p_value"] < 0.05


@pytest.mark.xfail(
    strict=True,
    reason="Raw EEG: the strongest periodogram peak is slow drift (periods of seconds) for most subjects; "
    "only 18% have it in the alpha band.",
)
def test_eyes_closed_rhythm_is_alpha(eeg):
    _, kept, _ = eeg
    closed = kept[kept["condition"] == "eyes_closed"].sort_values(["subject_id", "time"])
    periods = closed.groupby("subject_id")["oz_uv"].apply(lambda v: dominant_period(v.to_numpy(), detrend="linear"))
    assert periods.between(160 / 13, 160 / 8).mean() > 0.5
