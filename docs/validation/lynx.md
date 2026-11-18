# Validation of the population_dynamics profile: the Canadian lynx cycle

Roadmap item B3, first part ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
The Global Population Dynamics Database part (reddened spectra across many populations) comes later.

## Data

`data/fetch_population_series.py` writes the annual Canadian lynx trappings for 1821–1934 (114 years,
Mackenzie River district, Hudson's Bay Company records) to `data/external/lynx.csv`. This copy is R's
`datasets::lynx` (Brockwell & Davis, 1991, Series G), the series surveyed by Campbell & Walker
(1977), downloaded from the Rdatasets archive. The ten-year cycle in these returns was described by
Elton & Nicholson (1942).

## Profile

`population_dynamics` is a new profile. It computes `dominant_period`, `dominant_power_share`, spectral
entropy and sample entropy. Sample entropy is guarded by `min_samples: 100` (Richman & Moorman, 2000),
since many population records are 30–100 years long. Every metric is compared with 200 shuffled
copies of the counts (`surrogates: {type: shuffle}`).

## Criteria (fixed before running) and results

| criterion | result |
|---|---|
| 114 annual values, 1821–1934 | ✓ |
| dominant_period in [9, 11] years | **9.5** ✓ (114/12; the neighbouring possible periods are 10.4 and 8.8) |
| dominant power share above shuffles, p < 0.05 | 0.52 vs 0.08, p = 0.005 ✓ |
| spectral entropy below shuffles, p < 0.05 | 0.54 vs 0.89, p = 0.005 ✓ |

Not a criterion, but reported: sample entropy is 0.73 against 1.59 for shuffles (p = 0.005).
p = 0.005 is the smallest value possible with 200 surrogates.

## Notes

- Periods are whole fractions of the record length, so 9.5 and 10.4 are the two candidates near
  ten years. The criterion was set wide (9–11) for that reason.
- Raw counts were used, not log counts. The cycle's period is the same either way, but the power
  share depends on the scale. A log transform can be added with the `transforms` setting once that's merged (#84).

## References

- Brockwell, P. J. & Davis, R. A. (1991). *Time Series: Theory and Methods*, 2nd ed. Springer.
- Campbell, M. J. & Walker, A. M. (1977). A survey of statistical work on the Mackenzie River series of annual Canadian lynx trappings for the years 1821–1934 and a new analysis. *J. R. Stat. Soc. A* 140:411–431.
- Elton, C. & Nicholson, M. (1942). The ten-year cycle in numbers of the lynx in Canada. *J. Anim. Ecol.* 11:215–244.
- Richman, J. S. & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am. J. Physiol.* 278:H2039–H2049.
