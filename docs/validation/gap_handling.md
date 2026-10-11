# Validation of gap handling: synthetic data loss and GHCN

This page covers whether `gap_policy` handles gaps in a recording correctly. It checks the
policy against a published known answer on synthetic data, and against the gaps in real
temperature records. The settings and predictions in both pre-specified sections were fixed
before the first run.

## Policies

- `join` (default): a recording's samples are treated as consecutive, so values on either
  side of a gap become neighbours.
- `segment`: the recording is split at its gaps. DFA windows and sample-entropy templates sit
  inside one segment, and spectral entropy is a length-weighted mean over segments of at least
  16 samples. `max_joined_gap: N` joins gaps missing at most N samples and splits only at
  longer ones.
- `exclude`: recordings below `min_coverage` are dropped, and the rest are joined.

## Synthetic: Ma et al. (2010)

Ma et al. removed stretches from long-range correlated signals and joined what was left. For
positively correlated signals the DFA exponent was practically unchanged, even after 90%
loss. For anticorrelated signals it drifted toward 0.5.

**Setup:** fractional Gaussian noise with α = 0.3, 0.5, 0.7 and 0.9, 65,536 samples each.
Removed and kept stretches alternate with exponential lengths. Removed stretches average 200
samples, and 10% or 50% of the data is lost. DFA1 runs over windows of 16–512 samples, with 5
realizations per setting, and is compared with α on the same series before any loss.

| α | Loss | Before loss | `join` | `segment` |
|---|---|---|---|---|
| 0.3 | 10% | 0.306 | 0.308 | 0.309 |
| 0.3 | 50% | 0.305 | 0.317 | 0.311 |
| 0.5 | 10% | 0.499 | 0.494 | 0.495 |
| 0.5 | 50% | 0.507 | 0.511 | 0.507 |
| 0.7 | 10% | 0.711 | 0.712 | 0.716 |
| 0.7 | 50% | 0.702 | 0.702 | 0.688 |
| 0.9 | 10% | 0.892 | 0.894 | 0.899 |
| 0.9 | 50% | 0.908 | 0.925 | 0.919 |

| Prediction | Result |
|---|---|
| `segment` within 0.05 of the before-loss α for every α and loss | **Yes** (largest deviation 0.014) |
| `join` within 0.05 for α = 0.7 and 0.9 | **Yes** (largest deviation 0.017) |
| `join` raises α = 0.3 by at least 0.05 at 50% loss | **No** (+0.012), recorded as a strict `xfail` |

**Exploratory, after seeing the result:** the anticorrelated drift appears once the kept
stretches are much shorter than the largest window.

| Mean removed stretch (samples) | α = 0.3, 50% loss: `join` − before | `segment` |
|---|---|---|
| 200 (pre-specified) | +0.012 | 0.311 |
| 20 | +0.054 | NaN |
| 5 | +0.108 | NaN |

In those runs `segment` has no 512-sample window left inside any stretch, so it returns NaN.
That is the trade-off:

- `segment` is unbiased, but it can only see scales shorter than its segments.
- `join` sees every scale, but it is biased for anticorrelated signals when segments are short
  relative to the window.

Neither silently gives a wrong number for positively correlated signals.

## GHCN: all 23 stations

#57 excluded 5 of the 23 stations for coverage below 80%: Perm, Chita, Minsk, Brno and
Hong Kong. The gap report added in #69 then showed the included ones still contained gaps of
up to 11 years, joined silently (Feodosiya 4,019 days, Campbell Island 3,628 days).
`python data/fetch_ghcn_temperature.py --all-stations` writes all 23 stations.

**Pre-specified run:** `gap_policy: segment`, split at every gap, DFA2 over 30–3000 days.

| Prediction | Result |
|---|---|
| Islands higher, p < 0.05 | **No.** 0.719 ± 0.036 (n = 4) vs 0.679 ± 0.032 (n = 12), p = 0.11, d = 1.2 |
| Continental mean 0.65 ± 0.05, SD < 0.05 | **Yes.** 0.679, SD 0.032 |
| Chita (most continental) α ≥ 0.6, as Eichner et al. predict, not ≤ 0.55 as Fraedrich & Blender predict | **Yes.** 0.710 |

**Splitting at every gap is too aggressive for long windows.** 7 of 23 stations came out NaN,
because a single missing day splits a record. Oxford has 98% coverage, but scattered missing
days cut it into 1,246 segments, and fewer than four 3000-day windows fit. This led to
`max_joined_gap`.

**Exploratory, after seeing the result:** three policies on the same 23 stations, with the
joined-gap length set to the smallest DFA window (30 days) as a choice from the method rather
than from the results:

| Policy | Continental | Island | p |
|---|---|---|---|
| `join` | 0.687 ± 0.030 (n = 18) | 0.748 ± 0.060 (n = 5) | 0.083 |
| `segment` | 0.679 ± 0.032 (n = 12) | 0.719 ± 0.036 (n = 4) | 0.11 |
| `segment`, `max_joined_gap: 30` | 0.687 ± 0.030 (n = 16) | 0.749 ± 0.060 (n = 5) | 0.083 |

- **Station by station,** `join` and `segment` with `max_joined_gap: 30` agree within about
  0.01. That includes the stations with decade-long gaps (Feodosiya 0.720 vs 0.723, Campbell
  0.702 vs 0.699). For positively correlated temperature anomalies, joining across gaps doesn't
  bias DFA, consistent with Ma et al. (2010). The `climate_temperature` profile therefore keeps
  the default `join`.
- **The five stations added back** fall within the continental range (0.66–0.72 under `join`).
  With all 23 stations, the island difference is still in the published direction and still not
  significant.
- **Chita** is 0.71–0.72 under every policy. That is ordinary continental persistence, not the
  white-noise-like α ≈ 0.5 that Fraedrich & Blender (2003) report for inner continents.

## Reproducing

```bash
python data/fetch_ghcn_temperature.py --all-stations
python -m pytest -q tests/test_gap_validation.py tests/test_ghcn_validation.py
```

## References

- Davies, R. B. & Harte, D. S. (1987). Tests for Hurst effect. *Biometrika* 74(1):95–101.
- Eichner, J. F. et al. (2003). Power-law persistence and trends in the atmosphere. *Phys Rev E* 68:046133.
- Fraedrich, K. & Blender, R. (2003). Scaling of atmosphere and ocean temperature correlations in observations and climate models. *Phys Rev Lett* 90:108501.
- Ma, Q. D. Y., Bartsch, R. P., Bernaola-Galván, P., Yoneyama, M. & Ivanov, P. Ch. (2010). Effect of extreme data loss on long-range correlated and anticorrelated signals quantified by detrended fluctuation analysis. *Phys Rev E* 81:031101.
