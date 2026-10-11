# Validation on real climate data: GHCN-Daily temperatures

This page records what happened when the time-series pipeline was first run outside
biology, on real data with known published results. It covers what it reproduced, what it
didn't, and where the pipeline's assumptions didn't fit the data.

## Published result

Koscielny-Bunde et al. (1998) and Eichner et al. (2003) ran DFA on deseasonalized daily
maximum temperatures and found long-range power-law persistence above a crossover of about
10 days. Eichner et al. analyzed 95 stations:

- **Continental stations,** inland and coastal, have an exponent α close to 0.65 with a very
  narrow spread.
- **Island stations** range from 0.65 to 0.85, averaging about 0.8. The authors attribute
  this to coupling with the slowly varying ocean.

Fraedrich & Blender (2003) disagree for the continental interior: they report α ≈ 0.5 far
from the oceans.

## Data

The stations are the ones Eichner et al. name in their Figs. 1–3, matched to
[GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
(Menne et al., 2012). `data/fetch_ghcn_temperature.py` downloads them (~17 MB), keeps daily
maximum temperatures (TMAX) that passed all GHCN quality checks, and deseasonalizes each
station as Eichner et al. did, by subtracting the mean of each calendar date over all
years.

Inclusion rules were fixed before any exponent was computed. A station needs at least
12,000 valid days (enough for 3000-day windows) and at least 80% of the days between its
first and last value present. Wellington (NZ) has no long GHCN-Daily record, and the two
summit stations aren't part of the island-vs-continental contrast. Of the 23 remaining
stations, 18 qualified:

| Group | Included | Excluded (coverage) |
|---|---|---|
| Continental, inland | Vienna, Jena, Tashkent, Potsdam, Oxford, Cheyenne, Kunming | Perm (79%), Chita (76%), Minsk (76%), Brno (53%) |
| Continental, coastal | Charleston, Pusan, Anadyr, Hamburg, Plymouth, Feodosiya | Hong Kong (73%) |
| Island | Thursday Island, Koror, Raoul, Campbell, Wrangel | — |

Perm missed the coverage rule by one percentage point. The rule was kept as fixed. GHCN-Daily
records don't always match the paper's. Some cover different years, Plymouth's is a
different site (Mountbatten, from 1959), and Pusan's comes from the 1973 station.

## Method

```bash
python data/fetch_ghcn_temperature.py
python run_analysis.py --data data/external/ghcn_tmax_anomaly.csv --signal-cols tmax_anomaly \
  --condition-col condition --unit-cols station_id --domain climate_temperature \
  --output-dir outputs/ghcn
```

These settings were fixed in the `climate_temperature` profile before the run:

- **DFA2** (quadratic detrending of the profile), the order behind Eichner et al.'s summary
  statistics. It removes linear trends in the anomalies, such as urban warming.
- **Windows of 30–3000 days.** That starts about three times above the ~10-day crossover,
  and the shortest record (Thursday Island, 15,400 days) still holds four windows of the
  largest size.
- **Test:** Welch's t-test on one exponent per station, 13 continental vs 5 island.

## Results

| Measure | Continental | Island | p | Published | Reproduced? |
|---|---|---|---|---|---|
| DFA exponent α | 0.684 ± 0.032 | 0.748 ± 0.060 | 0.075 | ~0.65 (narrow) vs ~0.8 (0.65–0.85) | **Direction, size and spread: yes. Significance: no** |

Values are mean ± SD across stations. Cohen's d = 1.6.

| Station | Setting | α (DFA2) | Published (Eichner Figs. 1–2) |
|---|---|---|---|
| Vienna | inland | 0.68 | 0.64 ± 0.02 |
| Charleston | coastal | 0.67 | 0.63 ± 0.02 |
| Pusan | coastal | 0.69 | 0.67 ± 0.02 |
| Campbell Island | island | 0.70 | 0.71 ± 0.02 |
| Wrangel Island | island | 0.67 | 0.65 ± 0.02 |

**Continental stations** all show long-range persistence, from 0.63 (Cheyenne) to 0.74
(Plymouth), with an SD of 0.03. That matches the narrow, near-universal value reported. The
mean sits about 0.03 above 0.65, and so do the three named continental stations. The
exploratory scan below suggests why: 30 days is still close enough to the DFA2 crossover to
lift the exponent.

No continental station is near 0.5, so these results don't support Fraedrich & Blender's
inner-continental value. They don't test it well either: the station most deep inside a
continent, Chita, was excluded for coverage.

**Island stations** are higher on average and more varied, as published. Thursday Island
(0.82), Raoul (0.79) and Koror (0.75) are high. Campbell (0.70) and Wrangel (0.67) are close
to continental values, which is also what Eichner et al. found for those two stations.

**Significance.** The pre-specified two-sided test gives p = 0.075, so the difference is not
significant at 0.05. With only 5 island stations and their wide spread, the test is
underpowered. The result is recorded as a strict `xfail`, not as reproduced.

## Exploratory: sensitivity to DFA order and window

This scan was run *after* seeing the pre-specified result. It is reported in full and is
not used as validation:

| Settings | Continental | Island | p |
|---|---|---|---|
| DFA2, 30–3000 d (pre-specified) | 0.684 ± 0.032 | 0.748 ± 0.060 | 0.075 |
| DFA1, 30–3000 d | 0.669 ± 0.031 | 0.747 ± 0.064 | 0.051 |
| DFA3, 30–3000 d | 0.705 ± 0.034 | 0.747 ± 0.053 | 0.16 |
| DFA2, 10–1000 d | 0.780 ± 0.044 | 0.793 ± 0.055 | 0.66 |
| DFA2, 10–3000 d | 0.743 ± 0.036 | 0.772 ± 0.051 | 0.29 |
| DFA2, 30–1000 d | 0.710 ± 0.037 | 0.764 ± 0.063 | 0.13 |
| DFA2, 100–3000 d | 0.653 ± 0.033 | 0.757 ± 0.086 | 0.051 |
| DFA2, 365–3650 d | 0.625 ± 0.040 | 0.784 ± 0.162 | 0.093 |
| DFA2, 30–2000 d, 1951–1990 only | 0.673 ± 0.040 (n = 12) | 0.738 ± 0.075 | 0.13 |

Unlike the long-range heart-rate result in [fantasia.md](fantasia.md), the direction holds in
every setting: islands are always more persistent. Significance never reaches 0.05.

Windows that start at 10 days blur the contrast. They include the weather-regime range below
the crossover, where both groups look alike. Higher DFA orders raise the continental exponent
over a fixed window, because each order pushes DFA's crossover to larger windows (Eichner et
al., note 22). Starting at 100 days brings the continental mean to 0.653, the published value.
That range was not adopted here, since it was chosen after seeing the data.

## What the run found in the engine

The first non-biological run tested the claim that domain knowledge lives in the profile,
not in Python. It held only after two changes:

1. **Metric settings were hardcoded** (fixed). DFA scale ranges were baked into named
   metrics in `metrics.py` (`dfa_short_term` = 4–16 samples, the heart-rate convention),
   so a climate range would have meant adding Python. Profiles can now set
   `metric_parameters`, such as `dfa_exponent: {min_scale: 30, max_scale: 3000, order: 2}`.
   They're validated at load time (a misspelled parameter is an error, not silently ignored)
   and printed in the report.
2. **DFA detrending order was fixed at 1** (fixed). `dfa_exponent` now takes `order`
   (DFA1, DFA2, ...). DFA1 is unchanged and still matches the per-window `polyfit` reference
   exactly.

These assumptions didn't fit but were handled outside the engine for now. They are the input
for new data-shape design (#59) and unit handling (#35):

3. **No seasonal adjustment.** The pipeline has no step to remove a known periodic cycle, so
   the fetch script deseasonalizes. Any geophysical series with a daily, tidal, or annual
   cycle will need the same.
4. **Gaps are invisible.** The pipeline assumes regularly sampled series. Missing days are
   dropped and the remaining values joined, so DFA treats a 3-day gap as adjacent days. A
   coverage rule kept this small, but it excluded 5 of 23 stations, including the most
   continental one.
5. **Scales are in samples, not time.** `min_scale: 30` means 30 days only because there is one
   sample per day. The same profile on hourly data would silently mean 30 hours. That is a
   concrete case for #35.
6. **`date` isn't recognized as a time column.** Only `timestamp`, `time` and `timepoint` are,
   so the fetch script names the column `time`.

## Reproducing

```bash
python data/fetch_ghcn_temperature.py
python -m pytest -q tests/test_ghcn_validation.py
```

Continental persistence near 0.65 and the higher, more varied island exponents are asserted
as tests. The significance of the island difference is a strict `xfail`, so the suite fails
loudly if a future change makes it reproduce.

## References

- Eichner, J. F., Koscielny-Bunde, E., Bunde, A., Havlin, S. & Schellnhuber, H.-J. (2003). Power-law persistence and trends in the atmosphere: a detailed study of long temperature records. *Phys Rev E* 68:046133. [arXiv:physics/0212042](https://arxiv.org/abs/physics/0212042)
- Fraedrich, K. & Blender, R. (2003). Scaling of atmosphere and ocean temperature correlations in observations and climate models. *Phys Rev Lett* 90:108501.
- Koscielny-Bunde, E., Bunde, A., Havlin, S., Roman, H. E., Goldreich, Y. & Schellnhuber, H.-J. (1998). Indication of a universal persistence law governing atmospheric variability. *Phys Rev Lett* 81(3):729–732.
- Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E. & Houston, T. G. (2012). An overview of the Global Historical Climatology Network-Daily database. *J Atmos Oceanic Technol* 29(7):897–910.
