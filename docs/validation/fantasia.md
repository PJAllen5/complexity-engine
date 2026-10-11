# Validation on real heart-rate data: PhysioNet Fantasia

This page records what happened when the time-series pipeline was run on real data with
known published results. It covers what it reproduced, what it didn't, and what the run
exposed in the engine.

## Data

The [Fantasia database](https://physionet.org/content/fantasia/1.0.0/) (Iyengar et al.,
1996; Goldberger et al., 2000) contains 120-minute supine resting ECG recordings with
annotated heartbeats from 20 young (21–34 yr) and 20 elderly (68–85 yr) rigorously
screened healthy subjects. It is distributed under the Open Data Commons Attribution
License.

`data/fetch_fantasia.py` downloads the beat annotations and converts them to normal-to-normal
(NN) intervals. It keeps an interval only when both of its beats are annotated normal and
it lies between 0.3 and 2.0 s. That leaves 280,943 intervals, 4,731–9,179 per subject.
Exclusions were under 4% for 38 of 40 recordings. The other two, f2y09 and f2o08, lost 11%
and 18% to intervals next to non-normal beats.

## Method

The data was run through the unmodified CLI:

```bash
python data/fetch_fantasia.py
python run_analysis.py --data data/external/fantasia_rr.csv --signal-cols rr_interval \
  --condition-col condition --domain heart_rate_variability --output-dir outputs/fantasia
```

All settings were fixed before looking at the results:
- **Sample entropy:** m = 2, r = 0.2 SD.
- **DFA ranges:** 4–16 beats (short) and 16–64 beats (long), the standard heart-rate ranges
  (Peng et al., 1995).

Each subject is one recording. Groups are compared with Welch's t-test on one value per
subject, so n = 20 vs 20.

## Results

| Measure | Elderly | Young | p | Published | Reproduced? |
|---|---|---|---|---|---|
| Sample entropy | 1.25 ± 0.24 | 1.44 ± 0.16 | 0.006 | Lower in elderly | **Yes** |
| DFA, short range (4–16 beats) | 1.20 ± 0.27 | 1.05 ± 0.18 | 0.043 | Higher in elderly: 1.12 ± 0.19 vs 0.90 ± 0.14 | **Yes** (direction) |
| DFA, long range (16–64 beats) | 1.03 ± 0.12 | 1.01 ± 0.12 | 0.58 | Lower in elderly: 0.75 ± 0.17 vs 0.99 ± 0.10 | **No** |
| Mean RR interval (s) | 1.06 | 0.98 | 0.11 | — | — |

Values are mean ± SD across subjects. Published DFA values are from Iyengar et al. (1996),
on 10 young and 10 elderly Fantasia subjects.

**Sample entropy.** The direction matches reduced heart-rate entropy with age (Kaplan et
al., 1991). It also matches Balasubramanian & Nagaraj (2016), who used the same database
and settings and found elderly 1.13 ± 0.48 vs young 1.52 ± 0.43. Their values came from
100-beat segments, while these come from full recordings, so only the direction is
comparable. They also report that the direction reverses at r = 0.25 SD on short segments.
Sample entropy results are sensitive to their settings, so compare values only across
identical settings.

**Short-range DFA.** The direction and significance match. Both groups come out about
0.1–0.15 higher than the published values. Small windows bias DFA upward, and the exact
published fitting window is unknown, so a shift in absolute level is expected.

**Long-range DFA.** This was not reproduced over 16–64 beats. Iyengar et al.'s exact
long-range fitting window couldn't be confirmed from accessible sources. The
[abstract](https://pubmed.ncbi.nlm.nih.gov/8898003/) says only that elderly fluctuations
resemble white noise "over the longer range."

## Exploratory: the long-range result depends on the window

This scan was run *after* seeing the 16–64 beat result. It is reported in full and is not
used as validation:

| Window (beats) | Elderly | Young | p |
|---|---|---|---|
| 16–64 (pre-specified) | 1.03 ± 0.12 | 1.01 ± 0.12 | 0.58 |
| 32–128 | 0.84 ± 0.12 | 0.95 ± 0.08 | 0.002 |
| 64–256 | 0.74 ± 0.17 | 0.80 ± 0.09 | 0.16 |
| 100–1000 | 0.79 ± 0.13 | 0.70 ± 0.11 | 0.017 |
| 16–1000 | 0.84 ± 0.09 | 0.82 ± 0.06 | 0.58 |

At 32–128 beats the published pattern appears, with similar values. At 100–1000 beats it
reverses. A "long-range DFA exponent" is therefore not one quantity: the conclusion depends
on the window, and a report should always state it. Choosing 32–128 now would be fitting the
method to the answer. Settling this needs the published window. If it is confirmed,
`dfa_long_term` can be changed to match, and the result recorded as a test against that
published window.

## What the run found in the engine

Running on real data exposed three problems that synthetic data hadn't:

1. **DFA computed the fluctuation function incorrectly** (fixed). The engine averaged each
   window's RMS residual. Peng et al.'s definition takes one RMS over all residuals. The
   difference is negligible over the default 8 to n/4 range (white noise: 0.505 vs 0.498)
   but large at 4–16 samples (0.64 vs 0.59). The corrected version now matches `nolds` to
   within 0.02. This changes DFA values in every existing report slightly.
2. **A single DFA exponent mixes two regimes** (addressed). Over the default range, the
   engine reports elderly 0.89 vs young 0.85 (p = 0.045). That blends a short-range increase with the
   long-range behaviour and matches neither published finding. The new opt-in
   `dfa_short_term` and `dfa_long_term` metrics separate the two.
3. **Significance testing on raw samples was badly overconfident** (fixed in #64). The
   earlier test called the mean heart-rate difference significant at t = 119, p ≈ 0, by
   treating 281,000 beats as independent. Tested on one value per subject, it is
   p = 0.11, as expected for resting heart rate in healthy adults.

## Reproducing

```bash
python data/fetch_fantasia.py
python -m pytest -q tests/test_fantasia_validation.py
```

The sample entropy and short-range DFA findings are asserted as tests. The long-range
finding is a strict `xfail`, so the suite fails loudly, and the test should be updated,
if a future change makes it reproduce.

## References

- Balasubramanian, K. & Nagaraj, N. (2016). Aging and cardiovascular complexity: effect of the length of RR tachograms. *PeerJ* 4:e2755.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.
- Iyengar, N., Peng, C.-K., Morin, R., Goldberger, A. L. & Lipsitz, L. A. (1996). Age-related alterations in the fractal scaling of cardiac interbeat interval dynamics. *Am J Physiol* 271(4):R1078–R1084.
- Kaplan, D. T. et al. (1991). Aging and the complexity of cardiovascular dynamics. *Biophys J* 59(4):945–949.
- Peng, C.-K., Havlin, S., Stanley, H. E. & Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos* 5(1):82–87.
