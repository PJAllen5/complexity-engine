"""Known-answer validation of the eeg_oscillations profile on long resting EEG, roadmap item B1
(docs/design/validation-and-new-domains.md).

Published:
- Linkenkaer-Hansen et al. (2001, J Neurosci 21:1370): the alpha amplitude envelope has long-range
  temporal correlations; with eyes closed, EEG DFA exponents 0.68 +/- 0.07 over 5-300 s (10 subjects,
  20-min recordings); phase-randomized surrogates ~0.50.
- Montez et al. (2009, PNAS 106:1614): temporo-parietal alpha (6-13 Hz) DFA exponents over 1-25 s
  are lower in early Alzheimer's disease than in controls (0.66 vs 0.72, SEM 0.01, P < 0.005; MEG,
  19 vs 16, 4-min recordings).

Data: OpenNeuro ds004504 (Miltiadous et al., 2023), eyes-closed clinical EEG, unprocessed, decimated
to 125 Hz at fetch time; 29 healthy (12.5-16.5 min) and 36 Alzheimer's subjects.

Criteria, fixed before the data was downloaded:
(a) Healthy subjects, eeg_oscillations profile (O1/O2, 8-13 Hz envelope, DFA 5-150 s): the mean
    over subjects of each subject's O1/O2-averaged exponent is within 0.6-0.8, and at least 80%
    of the 58 healthy channel recordings (29 subjects x O1, O2) beat their 99 shuffles (p < 0.05).
(b) Alzheimer's vs healthy (T5, P3, Pz, P4, T6; 6-13 Hz envelope; DFA 1-25 s): each subject's
    exponent averaged over the five channels is lower in Alzheimer's on average, with Welch p < 0.05.

The data isn't committed (see data/fetch_resting_eeg.py), so this file is skipped unless the CSVs
exist under data/external/.

Result: (a) passes: 0.694 +/- 0.103 (published 0.68 +/- 0.07), shuffles 0.496, 88% of channels
beat them. (b) is in the published direction with values close to Montez's (0.665 vs 0.704;
published 0.66 vs 0.72) but p = 0.068 (strict xfail). See docs/validation/eeg_alpha_lrtc.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from complexity_engine.analysis import analyze_dataset

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
OCCIPITAL = EXTERNAL / "resting_eeg_occipital.csv"
TEMPOROPARIETAL = EXTERNAL / "resting_eeg_temporoparietal.csv"
CHANNELS = ["t5", "p3", "pz", "p4", "t6"]

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(not (OCCIPITAL.exists() and TEMPOROPARIETAL.exists()),
                                    reason="Resting EEG not downloaded; run python data/fetch_resting_eeg.py"),
]

MONTEZ_PROFILE = """alpha_montez:
  name: alpha_montez
  description: Temporo-parietal alpha envelope correlations as analyzed by Montez et al. (2009).
  recommended_metrics: [dfa_exponent]
  transforms:
""" + "".join(f'    {ch}: ["bandpass(6, 13)", envelope]\n' for ch in CHANNELS) + """  sampling_interval: 0.008s
  metric_parameters:
    dfa_exponent: {min_scale: 125, max_scale: 3125}
"""


@pytest.fixture(scope="module")
def healthy(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lrtc")
    data = pd.read_csv(OCCIPITAL)
    data[data["condition"] == "healthy"].to_csv(folder / "healthy.csv", index=False)
    result = analyze_dataset(folder / "healthy.csv", signal_cols=["o1", "o2"], unit_cols=["subject_id"],
                             domain="eeg_oscillations", output_dir=folder / "out")
    return result["complexity_by_unit"]


@pytest.fixture(scope="module")
def montez(tmp_path_factory):
    folder = tmp_path_factory.mktemp("montez")
    (folder / "profile.yaml").write_text(MONTEZ_PROFILE, encoding="utf-8")
    result = analyze_dataset(TEMPOROPARIETAL, signal_cols=CHANNELS, unit_cols=["subject_id"], domain="alpha_montez",
                             domain_config=folder / "profile.yaml", output_dir=folder / "out")
    units = result["complexity_by_unit"]
    return units.groupby(["condition", "subject_id"])["dfa_exponent"].mean()


def test_healthy_alpha_shows_long_range_correlations(healthy):
    per_subject = healthy.groupby("subject_id")["dfa_exponent"].mean()
    assert len(per_subject) == 29
    assert 0.6 <= per_subject.mean() <= 0.8


def test_healthy_envelopes_beat_their_shuffles(healthy):
    assert (healthy["dfa_exponent_surrogate_p"] < 0.05).mean() >= 0.8


@pytest.mark.xfail(
    strict=True,
    reason="Alzheimer's 0.665 vs healthy 0.704 (published 0.66 vs 0.72), but Welch p = 0.068. Scalp EEG "
    "referenced to Cz is noisier than Montez's MEG gradiometers, which widens the spread.",
)
def test_alzheimers_alpha_correlations_are_weaker(montez):
    alzheimers, controls = montez["alzheimers"], montez["healthy"]
    assert (len(alzheimers), len(controls)) == (36, 29)
    result = stats.ttest_ind(alzheimers, controls, equal_var=False)
    assert alzheimers.mean() < controls.mean() and result.pvalue < 0.05
