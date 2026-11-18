# Validation on WESAD: heart rate and skin conductance under acute stress

Roadmap item A4 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This is the first real-data check for the `stress` profile, and the roadmap's weakest known
answer: a direction, not a published complexity value.

## Data

WESAD (Schmidt et al., 2018) covers 15 subjects with a chest RespiBAN sampled at 700 Hz. The study
recorded a 20-minute baseline and the Trier Social Stress Test (TSST; Kirschbaum et al., 1993) among
other conditions. It's licensed for academic, non-commercial use, so you download `WESAD.zip`
(2.25 GB) yourself. `data/fetch_wesad.py` then:

- reads only the archive's plain-text files: `SX_respiban.txt` and `SX_quest.csv`, streamed one
  subject at a time. It never loads the `SX.pkl` pickles, because unpickling a downloaded file can
  run code.
- cuts each condition using the protocol times, given as minutes.seconds. Seconds under 10 are
  written with a leading zero (`26.04`), so a one-digit fraction is a dropped trailing zero:
  `50.3` is 50 min 30 s.
- converts ECG and EDA with the formulas in WESAD's readme.
- detects R-peaks with a 5–15 Hz band-pass, squared slope, and peaks at least 0.33 s apart. RR
  intervals outside 0.3–2.0 s, or more than 20% from their local median, are dropped (median 0.8%).
- bins skin conductance to 4 Hz.

## Criteria (fixed before running) and results

| criterion | result |
|---|---|
| 15 subjects with baseline and stress | ✓ |
| ≥ 95% of RR intervals within 0.4–1.5 s (no reference annotations exist) | 99.6% of 37,636 ✓ |
| heart rate higher under stress, per-recording p < 0.05 | 73.0 → 99.3 bpm, p < 0.001, d = 1.7; all 15 subjects ✓ |
| skin conductance higher under stress, per-recording p < 0.05 | 3.91 → 5.96 µS, **p = 0.137** ✗ (strict xfail) |

**Exploratory, with the hypothesis stated in advance:** heart-rate sample entropy is lower under
stress, 1.38 → 0.99 (p = 0.004). Spectral entropy falls too (0.74 → 0.62) and the DFA exponent
rises (0.88 → 0.98). Faster, more regular heartbeats under stress fit the reduced vagal control that
the stress profile's interpretation describes. These weren't pass/fail criteria.

## Why skin conductance failed

13 of 15 subjects have higher skin conductance under stress. But resting skin conductance differs
a lot between people (some baselines are near 1 µS, others near 10). The engine's per-recording
test compares the stress recordings with the baseline recordings as two independent groups. It
ignores that they are the same 15 people, which its report already calls conservative. A paired
comparison (each subject's change) would remove the between-person spread. That's a reasonable
engine feature to add, but it wasn't part of this validation, and adding it to rescue this result
would be choosing the test after the fact.

## References

- Kirschbaum, C., Pirke, K.-M. & Hellhammer, D. H. (1993). The 'Trier Social Stress Test': a tool for investigating psychobiological stress responses in a laboratory setting. *Neuropsychobiology* 28:76–81.
- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C. & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proc. ICMI 2018*, 400–408.
