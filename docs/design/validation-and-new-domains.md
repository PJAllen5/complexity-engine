# Roadmap: real-data validation for every profile, and five new domains

Status: proposal, 2026-11-17. Follows the pattern of #57/#58: every profile gets a real dataset
with a published answer, settings and pass criteria fixed **before** the run, a fetch script that
keeps the data out of git, a skip-if-missing test, and a `docs/validation/<name>.md` write-up.

Today 3 of 12 profiles are validated on real data (`heart_rate_variability`,
`climate_temperature`, `social_network`). This plan covers the other 9 and adds 5 new domains:
EEG, finance, ecology population counts, gait, and traffic/infrastructure networks.

---

## Part 0 — Engine prerequisites

Several validations need engine features that don't exist yet. Each is opt-in and leaves
existing outputs unchanged, like the time-axis work.

| # | Feature | Needed by | Notes |
|---|---|---|---|
| E1 | **Signal transforms** in profiles: `transforms: {price: [log_return]}` with `diff`, `log_return`, `abs`, `zscore`, `bandpass(lo,hi)`, `envelope` (Hilbert amplitude), `resample(interval)` | finance, EEG, attention | Transforms run per recording unit, before metrics. The report lists each transform applied, so a derived signal is never mistaken for the raw one. |
| E2 | **Surrogate baselines**: shuffled and phase-randomized surrogates for DFA and entropy, reported as `metric_surrogate_mean` and an empirical p | EEG, attention, finance, neural | Matches the engine's rule of never quoting a statistic without its null. Seeded; count configurable (default 200). |
| E3 | ~~Design PR 3~~ `axis: time` and duration scales (`30D`, `5s`) | EEG (seconds), ecology (years), finance (trading days) | **Done** in #73; listed for completeness. |
| E4 | **Short-series guardrails**: a minimum length per metric (e.g. DFA needs ≥ 4 × `max_scale`; SampEn needs ≥ ~10^m … 30^m points), NaN with a reason otherwise | ecology | Annual counts are often 30–100 points long. Today a short series yields a number that looks trustworthy. |
| E5 | **`dominant_period` metric**: the period of the strongest spectral peak, with its share of total power | ecology, EEG alpha peak | Fits next to spectral entropy in `metrics.py`. |
| E6 | **Large-network mode**: sampled average shortest path (k random sources), skip sigma/omega above N nodes with a reason, timing test | traffic/infrastructure | roadNet-CA has ~2M nodes; exact all-pairs paths are infeasible. |
| E7 | **Fetch-time format conversion** (WFDB, EDF, NWB → CSV) in `data/fetch_*.py`, with a `[fetch]` optional dependency group | gait, EEG, neural | Keeps the engine itself CSV/Excel-only. |

Also needed by Part A: **design PR 5 (attribute mixing)** for the dolphin network's
known-group check. It builds on the node table (design PR 4, branch `add-node-table`).

---

## Part A — Validating the 9 unvalidated profiles

For each: dataset · known answer · pass criterion · engine work · new references.

### A1. `neuronal_network` — *C. elegans* connectome (quick win)
- **Data:** the *C. elegans* neural network used by Watts & Strogatz (1998). Public versions
  differ slightly in node count (282 in the paper, 297 in Newman's `celegansneural`). Fix which
  version and how to symmetrize it before running.
- **Known answer:** Watts & Strogatz report L = 2.65 vs 2.25 for a random graph, and C = 0.28 vs 0.05.
- **Pass:** clustering is several times the degree-matched random value, path length is within
  ~20% of random, and sigma > 1 (`--compute-small-world`). Tolerances fixed in advance because of
  the node-count difference.
- **Engine work:** none. **Refs:** Watts & Strogatz 1998 (already in bib); White et al. 1986.

### A2. `network` — US western power grid (quick win)
- **Data:** the power-grid network from Watts & Strogatz (1998), 4,941 nodes (Newman's network data).
- **Known answer:** L = 18.7 vs 12.4 random; C = 0.080 vs 0.005 random.
- **Pass:** exact node count; C and L within tolerance of the published values; small-world sigma > 1.
- **Engine work:** none. This also seeds the traffic/infrastructure domain (B5).

### A3. `animal_social_network` — Doubtful Sound dolphins (quick win)
- **Data:** Lusseau's dolphin association network: 62 dolphins, 159 ties, undirected.
- **Known answer:** an exact node and edge count. Newman & Girvan (2004) split it into two
  large groups with modularity Q = 0.52. The split follows the real fission observed when one
  dolphin left the group.
- **Pass:** 62 / 159 exactly; modularity ≥ ~0.5; the two largest detected communities match the
  known split, scored with `community_nmi` (needs PR 5) and a misassignment count, as with the
  karate club.
- **Engine work:** PR 5 for NMI, or a misassignment-count fallback. **Refs:** Lusseau 2003;
  Lusseau et al. 2003; Newman & Girvan 2004; Lusseau & Newman 2004 (already cited).

### A4. `physiology` and `stress` — WESAD
- **Data:** WESAD (Schmidt et al. 2018). 15 subjects, chest ECG/EDA/respiration at 700 Hz and wrist
  sensors, in baseline, stress (Trier Social Stress Test) and amusement conditions. Licensed for
  academic, non-commercial use: the fetch script should print the licence.
- **Known answer:** a *directional* check, not a published complexity value. The TSST reliably
  raises heart rate and skin conductance, so `signal_mean` for HR and EDA should be higher under
  stress than baseline, with ≥ 2 recordings per condition using the per-recording test.
  Complexity metrics (lower HRV entropy under stress) are reported as exploratory, with the
  hypothesis stated in advance.
- **Pass:** HR and EDA stress > baseline, p < 0.05 across subjects.
- **Engine work:** E7 (derive RR intervals from ECG at fetch time, downsample EDA).
  **Refs:** Schmidt et al. 2018; Kirschbaum et al. 1993 (TSST).
- **Caveat to note in the doc:** this is the weakest known answer in the plan. It validates the
  multi-signal condition pipeline, not a published complexity number.

### A5. `cognitive` — EEG eyes open vs eyes closed
- **Data:** PhysioNet EEG Motor Movement/Imagery Database: 109 subjects, 64 channels, 160 Hz.
  Runs 1 and 2 are one-minute eyes-open and eyes-closed baselines.
- **Known answer:** the Berger effect. Closing the eyes boosts occipital alpha (8–13 Hz), which
  concentrates spectral power and lowers spectral entropy at O1/Oz/O2.
- **Pass:** spectral entropy is lower with eyes closed in a clear majority of subjects, with
  p < 0.05 per-recording; `dominant_period` falls in the alpha range with eyes closed.
- **Engine work:** E5, E7 (EDF → CSV). **Refs:** Schalk et al. 2004; Goldberger et al. 2000.

### A6. `attention` — reaction-time series
- **Data:** to be chosen. Criteria: ≥ 500 consecutive trials per subject from a simple
  RT/tapping task, public, with a shuffled-order control. If none fits, collecting data (e.g.
  a web task with club volunteers) needs IRB approval first.
- **Known answer:** human RT and production series show 1/f-like long-range correlation
  (Gilden et al. 1995; Van Orden et al. 2003): DFA α above 0.5, short of 1.
- **Pass:** α > shuffled-surrogate α (E2) in most subjects; median α in (0.5, 1.0).
- **Engine work:** E2. **Refs:** Gilden et al. 1995; Van Orden et al. 2003.

### A7. `neural_activity` — synthetic known answer, then Allen Brain Observatory
- **Synthetic (known answer):** simulate spike trains, then convolve them with a GCaMP6 decay kernel.
  (a) Poisson spikes → α ≈ 0.5 above the decay scale. (b) Spikes with a long-range-correlated
  rate → α clearly above 0.5. (c) Calcium smoothing lowers SampEn relative to the spike train.
  This tests the profile's own cautions.
- **Real (pre-stated consistency check):** Allen Brain Observatory Visual Coding 2-photon ΔF/F,
  spontaneous epochs vs drifting gratings. Hypotheses are written before running, as for the GHCN rerun.
- **Engine work:** E2, E7 (NWB → CSV). **Refs:** Chen et al. 2013 (already cited); de Vries et al. 2020.

### A8. `collective_behavior` — Vicsek model, then real trajectories
- **Synthetic (known answer):** the Vicsek et al. (1995) model across noise levels. Polarization
  is near 1 at low noise and near 0 at high noise, with a transition between. Near the
  transition, polarization fluctuations should show the most temporal structure (highest α).
- **Real:** published fish-school trajectories (e.g. idtracker.ai datasets, or Tunstrøm et al.
  2013 if available). Known answer: schools switch between polarized and milling states
  (bimodal polarization/rotation). Also run behavetrack on the same videos and check it
  reproduces the published trajectories' metrics.
- **Engine work:** none for synthetic; E7 for real data. **Refs:** Vicsek 1995; Tunstrøm 2013
  (already cited); Romero-Ferrero et al. 2019.

### A9. Behavetrack end-to-end
Once A7/A8 pass on the published trajectories, run `behavetrack export-complexity` on raw video
for the same experiment. The engine's output from behavetrack data should match its output from
the published trajectories, within tolerance. This validates the two-tool chain, not just the engine.

---

## Part B — New domains

Each new domain = profile in `config/domain_profiles.yaml` + bib entries + fetch script +
validation test + doc. These are config work once Part 0 is in place.

### B1. `eeg_oscillations` — long-range temporal correlations in alpha
- **Profile:** `transforms: [bandpass(8,13), envelope]`, `axis: time`, DFA scales 5 s–300 s,
  surrogate baseline on.
- **Known answer:** Linkenkaer-Hansen et al. (2001) found power-law scaling in the alpha/beta
  amplitude envelope over 5–300 s. DFA exponents were about 0.68 (eyes closed) and 0.70 (eyes
  open), well above surrogates.
- **Data:** needs ≥ 10 min of continuous rest. EEGMMIDB's one-minute runs are too short. A
  candidate long resting-state dataset (e.g. from OpenNeuro) is to be chosen, with the length
  criterion fixed first.
- **Pass:** α in roughly 0.6–0.8 and above the surrogate in most subjects.
- **Engine work:** E1, E2 (E3 done).

### B2. `financial_returns` — stylized facts of markets
- **Profile:** `signal_roles` for price; `transforms: {price: [log_return]}` and a second
  derived signal `[log_return, abs]` (volatility proxy); DFA on both.
- **Known answer:** returns are nearly uncorrelated (α ≈ 0.5). Volatility is long-range
  correlated: Liu et al. (1999) report α₂ = 0.93 for S&P 500 volatility at scales beyond
  ~1.5 days (up to a year), and α₁ = 0.66 below that.
- **Data:** long daily S&P 500 index series. FRED's `SP500` covers only ~10 years, so pick a
  longer free source and check its terms. Trading-day gaps: use `axis: samples`, so weekends
  aren't gaps.
- **Pass:** α(returns) within 0.5 ± 0.1; α(|returns|) clearly > α(returns) and near 0.9 at long
  scales; both compared with shuffled surrogates.
- **Engine work:** E1, E2. **Refs:** Liu et al. 1999; Cont 2001.

### B3. `population_dynamics` — ecology population counts
- **Profile:** short-series aware; `dominant_period` and spectral metrics; DFA only where long enough.
- **Known answers:** (1) The Hudson's Bay Company Canadian lynx returns (1821–1934, 114 years)
  show the classic ~10-year cycle (Elton & Nicholson 1942): `dominant_period` ≈ 9–10 years,
  low spectral entropy. (2) Across many populations, spectra are "reddened" (more
  low-frequency variance) (Inchausti & Halley 2001, Global Population Dynamics Database).
- **Pass:** lynx period within 9–11 years; most GPDD series have a positive spectral exponent,
  with the subset and length cutoff fixed beforehand.
- **Engine work:** E4, E5 (E3 done). **Refs:** Elton & Nicholson 1942; Inchausti & Halley 2001.

### B4. `gait` — stride-interval dynamics
- **Profile:** stride interval as the signal, `axis: samples` (scales in strides, like HRV in
  beats), DFA over 16–N strides.
- **Known answer:** Hausdorff et al. (1997) found α lower in healthy elderly than young adults
  (0.68 ± 0.14 vs 0.87), and lower in Huntington's disease than controls (0.60 ± 0.24 vs
  0.88 ± 0.17, p < 0.005).
- **Data:** PhysioNet gait databases (Gait Dynamics in Neuro-Degenerative Disease; gait in aging).
  Confirm which cohorts match the 1997 paper. Pass on direction and significance, not exact
  values, if the cohorts differ.
- **Pass:** α(HD) < α(control) and α(elderly) < α(young), p < 0.05; control α near 0.9.
- **Engine work:** E7 only. This is the cheapest new domain. **Refs:** Hausdorff et al. 1995, 1997.

### B5. `infrastructure_network` — power, roads, transport
- **Profile:** network pipeline; interpretation for robustness, redundancy and spatial
  constraint (low clustering, long paths in road networks).
- **Known answers:** the power grid (A2), plus SNAP `roadNet-CA` published size statistics
  (exact node/edge counts, largest component) as a scale test.
- **Pass:** A2's criteria; roadNet-CA counts exact, a sampled path length that stays stable
  across seeds, runtime under a fixed budget.
- **Engine work:** E6. Traffic *flow* time series (e.g. PeMS) can follow later as a time-series
  profile, but it needs an account and is out of scope here.

---

## Delivery order

Each step is one PR (or a small group) that leaves existing outputs unchanged.

**Phase 1 — quick wins, no engine changes**
1. A1 *C. elegans* + A2 power grid (one `fetch_reference_networks.py`, one test file)
2. A3 dolphins (misassignment count now; NMI once PR 5 lands)
3. B4 gait (E7 for WFDB stride files) — first new domain

**Phase 2 — engine prerequisites**
4. E2 surrogate baselines
5. E1 signal transforms
6. E4 short-series guardrails + E5 `dominant_period`
7. Design PR 5 (attribute mixing); then upgrade A3 to NMI

**Phase 3 — validations that need Phase 2**
8. A5 EEG eyes open/closed (`cognitive`)
9. B1 EEG alpha LRTC
10. B2 finance
11. A4 WESAD (`physiology`, `stress`)
12. A6 attention (once a dataset is chosen)
13. B3 ecology (lynx, then GPDD)

**Phase 4 — synthetic-first domains and scale**
14. A7 neural activity (synthetic → Allen)
15. A8 collective behavior (Vicsek → real trajectories), then A9 behavetrack end-to-end
16. E6 large-network mode + B5 roadNet-CA

**After each validation:** update the README's "Validation on real data" section, add the bib
entries (the citation-integrity test will catch missing ones), and record a pass/partial/fail
honestly. Fantasia's unreproduced long-range DFA result is the model for reporting a partial pass.

## Open decisions

1. **Attention dataset:** find a public one, or collect one (needs IRB)?
2. **Surrogate count default** (E2): 200 is fast. Publication-grade p-values may want 1,000.
3. **Licences:** WESAD is non-commercial; finance and GPDD sources need checking. Fetch scripts
   should print terms, and nothing should be committed.
4. **EEG and finance sources:** pick specific long-recording datasets before writing the criteria.

## References to add to `papers/references.bib`

Cont 2001 · Elton & Nicholson 1942 · Gilden, Thornton & Mallon 1995 · Goldberger et al. 2000 ·
Hausdorff et al. 1995, 1997 · Inchausti & Halley 2001 · Kirschbaum, Pirke & Hellhammer 1993 ·
Linkenkaer-Hansen et al. 2001 (check if present) · Liu et al. 1999 · Lusseau 2003 ·
Lusseau et al. 2003 · Newman & Girvan 2004 · Romero-Ferrero et al. 2019 · Schalk et al. 2004 ·
Schmidt et al. 2018 · Van Orden, Holden & Turvey 2003 · Vicsek et al. 1995 (check if present) ·
de Vries et al. 2020 · White et al. 1986
