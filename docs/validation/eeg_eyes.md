# Validation on resting EEG: eyes open vs eyes closed

Roadmap item A5 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This is the first real-data check for the `cognitive` profile.

## Data

`data/fetch_eeg_eyes.py` downloads runs 1 (eyes open) and 2 (eyes closed) for all 109 subjects of
PhysioNet's EEG Motor Movement/Imagery Database (Schalk et al., 2004; Goldberger et al., 2000). Each
run is one minute of 64-channel EEG at 160 Hz. The script reads the EDF files itself (roadmap E7),
keeps the Oz channel (central occipital), and deletes each EDF after conversion. All 218 recordings
are at 160 Hz. The signals are raw, as recorded by BCI2000: no filtering.

## Criteria (fixed before running) and results

Known answer: the Berger effect. Closing the eyes boosts occipital alpha (8–13 Hz).

| criterion | result |
|---|---|
| recordings not at 160 Hz excluded and counted | none to exclude ✓ |
| spectral entropy lower with eyes closed in ≥ 70% of subjects, Welch p < 0.05 | **42%; means 0.651 closed vs 0.635 open ✗** |
| eyes-closed dominant period in the alpha band in > 50% of subjects | **18% ✗** |

## Why it failed

The Berger effect is plainly present in the recordings:

| median share of power | eyes open | eyes closed |
|---|---|---|
| alpha, 8–13 Hz | 0.072 | 0.235 |
| below 1 Hz | 0.451 | 0.312 |
| 59–61 Hz (mains) | 0.006 | 0.003 |

Alpha is higher with eyes closed in 92% of subjects. That comparison was chosen after the failure,
so it isn't counted as a pass. But it shows what went wrong. Most of the raw signal's power is
slow drift below 1 Hz, and there is more of it with eyes open, likely from eye movements and blinks.
That drift concentrates the eyes-open spectrum, lowering its spectral entropy, and it is the strongest
"rhythm" for most subjects.

**Lesson for the `cognitive` profile:** spectral entropy and dominant period computed on unfiltered
EEG mostly measure artefacts and drift. Band-limit the signal first, which the `transforms` setting
(#84) will allow.

## Criteria for the band-passed rerun (fixed now, before it has been run)

Once #84 is merged, the same data will be rerun with `transforms: {oz_uv: ["bandpass(1, 40)"]}` and
`sampling_interval: 0.00625s`. Within 1–40 Hz, drift and mains noise are removed and alpha should
dominate with eyes closed. The criteria:

1. Spectral entropy lower with eyes closed in ≥ 70% of subjects, and the per-recording Welch test
   gives p < 0.05 with the closed mean lower.
2. With eyes closed, dominant period in the alpha band (12.3–20 samples) in > 50% of subjects.

## References

- Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. *Archiv für Psychiatrie und Nervenkrankheiten* 87:527–570.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101:e215–e220.
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. & Wolpaw, J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Trans. Biomed. Eng.* 51:1034–1043.
