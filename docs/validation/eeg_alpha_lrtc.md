# Validation of the eeg_oscillations profile: long-range correlations in resting alpha

Roadmap item B1 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This is the first validation that uses signal transforms (#84): band-pass to alpha, then the
amplitude envelope.

## Data

OpenNeuro ds004504 (Miltiadous et al., 2023, CC0) is routine clinical EEG with eyes closed at
rest: 19 channels at 500 Hz, referenced to Cz. It was the only open dataset found with at least
10 minutes of continuous eyes-closed rest. The groups used are 29 healthy older adults (12.5–16.5
min) and 36 people with Alzheimer's disease. `data/fetch_resting_eeg.py`:
- reads the **unprocessed** EEGLAB files with `scipy.io.loadmat`. The dataset's cleaned
  derivatives cut out bad stretches, which would add hidden gaps.
- keeps O1, O2, T5, P3, Pz, P4 and T6.
- decimates to 125 Hz with a zero-phase anti-aliasing filter.
- deletes each file after conversion. That's about 2 GB downloaded and 660 MB of CSV kept, so this
  validation isn't in CI.

## Published answers (checked against the papers' full text)

| | Linkenkaer-Hansen et al. (2001) | Montez et al. (2009) |
|---|---|---|
| recording | EEG + MEG, eyes closed, 10 young adults, 20 min | MEG, 19 early AD vs 16 controls, 4 min |
| band | individual alpha peak (~8.3–11.7 Hz) | 6–13 Hz |
| DFA window | 5–300 s | 1–25 s |
| result | EEG α = 0.68 ± 0.07; surrogates ≈ 0.50 | AD 0.66 vs controls 0.72 (SEM 0.01), P < 0.005 |

## Criteria (fixed before the data was downloaded) and results

**(a) Healthy subjects.** `eeg_oscillations` profile on O1/O2, 8–13 Hz envelope, DFA 5–150 s.
150 s rather than 300 s, because these records are 12.5–16.5 min and DFA needs four windows of
the largest scale.

| criterion | result |
|---|---|
| subject mean α within 0.6–0.8 | **0.694** ± 0.103 (range 0.51–0.91) ✓ |
| ≥ 80% of channel recordings beat their 99 shuffles (p < 0.05) | 88% (shuffles average 0.496) ✓ |

**(b) Alzheimer's vs healthy.** Temporo-parietal T5/P3/Pz/P4/T6, 6–13 Hz envelope, DFA 1–25 s,
averaged over channels per subject.

| criterion | result |
|---|---|
| AD lower, Welch p < 0.05 | 0.665 vs 0.704, **p = 0.068** ✗ (strict xfail) |

The direction matches, and the values are close to the published ones: 0.665 vs 0.66 for AD, and
0.704 vs 0.72 for controls. But the difference isn't significant here. The likely reason is the
recording: clinical scalp EEG referenced to Cz is much noisier than the MEG planar gradiometers
Montez et al. used, so the spread between subjects is larger. The AD group here is also wider
(early and later stages, with MMSE 17.8 ± 4.5) than "early-stage".

## References

- Linkenkaer-Hansen, K., Nikouline, V. V., Palva, J. M. & Ilmoniemi, R. J. (2001). Long-range temporal correlations and scaling behavior in human brain oscillations. *J. Neurosci.* 21:1370–1377.
- Miltiadous, A. et al. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data* 8:95.
- Montez, T. et al. (2009). Altered temporal correlations in parietal alpha and prefrontal theta oscillations in early-stage Alzheimer disease. *PNAS* 106:1614–1619.
