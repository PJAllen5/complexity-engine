# Complexity Engine

A small, reproducible Python toolkit for domain-aware complexity analysis of scientific data.
It has two parallel pipelines for two different data shapes:

- a **time-series pipeline** (`run_analysis.py`) for tabular signal/condition data
- a **network pipeline** (`run_network_analysis.py`) for graph-shaped edge-list data

Both apply a domain-specific set of complexity or network-science metrics and generate a
report that explains the results in the language of the chosen scientific domain, grounded in
the project's literature base.

The project started as a literature review workspace for complexity science and information
theory (see `papers/references.bib`) and has since grown into these working analysis tools
built on that foundation.

## Time-series pipeline: what it does today

1. **Ingest** one or more CSV/Excel files of time-series data (one or more signal columns, a
   condition/group column, and optional metadata like subject ID or trial). Multiple files
   (e.g. one per session) are merged if they share the same columns — a mismatch raises a
   clear error rather than silently merging incompatible data.
2. **Clean**: drops rows missing a required value (signal or condition), and detects — but
   does not automatically remove — exact duplicate rows, since two different time points can
   legitimately share the same signal value. Every input/dropped/output row count is recorded
   and surfaced in a "Data provenance" report section, so cleaning is never silent.
3. **Summarize** each signal by condition (mean, std, min, max, median).
4. **Compute complexity metrics** per condition — sample entropy, spectral entropy, and a DFA
   (detrended fluctuation analysis) scaling exponent, plus opt-in short-range (4–16 samples)
   and long-range (16–64 samples) DFA exponents (`dfa_short_term`, `dfa_long_term`) for
   signals whose correlation structure changes with scale, and opt-in `dominant_period` (the
   period, in samples, of the strongest spectral peak, at most half the series) with
   `dominant_power_share` (that peak's share of the power) for rhythmic signals — restricted to whichever metrics
   the active domain profile recommends. These metrics depend on sample order, so they're computed
   separately within each independent recording (subject, trial, session — auto-detected from
   `subject_id`/`subject`/`participant`/`session`/`trial`, or set with `--unit-cols`), sorted by
   a `timestamp`/`time`/`timepoint`/`date` column when present, then averaged across recordings
   per condition. Joining recordings end to end would insert artificial jumps that the metrics
   would read as real dynamics. Per-recording values are written to `complexity_by_unit.csv`.
   With a time column, each recording's sampling is also checked and reported
   (`sampling_by_unit.csv`): the interval, coverage, gaps (spacings over 1.5 intervals), and
   whether the spacing is irregular. A profile's `gap_policy` sets how metrics treat gaps:
   `join` (default) treats samples as consecutive and reports how many gaps were joined;
   `segment` computes within gap-free segments so no DFA window or entropy template spans a
   gap (`max_joined_gap: N` joins gaps of at most N missing samples); `exclude` drops
   recordings below `min_coverage`. Joining is sound for positively correlated signals but can
   bias anticorrelated ones (see [docs/validation/gap_handling.md](docs/validation/gap_handling.md)).
5. **Test statistical significance** between every pair of conditions (Welch's t-test +
   Cohen's d effect size), so directional differences can be read alongside whether they're
   distinguishable from sampling noise. Time-series samples are autocorrelated and nested
   within recordings, so they're never tested as independent observations: when each
   condition has at least two recordings, the test compares one value per recording — its
   mean signal level and each complexity metric. With a single recording per condition, it
   falls back to the raw samples with each recording's sample size reduced to its AR(1)
   effective sample size (Zwiers & von Storch, 1995), which the report flags as approximate.
6. **Compute cross-signal coupling**: when more than one signal column is selected, pairwise
   mutual information is computed per condition, capturing how tightly signals are coupled.
7. **Interpret domain-aware**: each domain profile supplies a guiding analysis question,
   descriptions of signal roles (primary signal / condition / covariate), and
   metric-specific interpretation text (including for cross-signal coupling). The generated
   report uses these to explain *why* a higher or lower value matters for that domain, rather
   than just printing numbers. With multiple signals, the interpretation is grouped into a
   subsection per signal plus a cross-signal relationships subsection.
8. **Visualize**: generates a condition-comparison box plot and a complexity-metrics bar
   chart per signal, plus a mutual-information heatmap when multiple signals are selected —
   saved as PNGs under `outputs/plots/` and embedded in `report.md`.
9. **Report**: writes `condition_summary.csv`, `complexity_summary.csv`, `complexity_by_unit.csv`,
   `sampling_by_unit.csv` (when there's a time column), `significance_summary.csv`, `cross_signal_summary.csv`, the plot PNGs, a human-readable
   `report.md`, and a self-contained `report.html` (same content, styled, with the plots
   base64-inlined into the single file) to an output directory.

Domain profiles for this pipeline (`physiology`, `cognitive`, `stress`, `attention`,
`heart_rate_variability`, `climate_temperature`, `gait`, `population_dynamics`, and any
custom ones you add) live in `config/domain_profiles.yaml` (or `.json`) and can be extended or
overridden without touching Python code. A profile's `recommended_metrics` is validated
against the pipeline's known metrics as soon as the domain is looked up — before any data is
loaded — so a typo or a metric name that belongs to the other pipeline fails immediately with
a clear message, not deep inside a later analysis step.

A profile can also set `metric_parameters` to override a metric's settings for its domain,
for example a DFA window range and detrending order:

```yaml
metric_parameters:
  dfa_exponent: {min_scale: 30, max_scale: 3000, order: 2}  # scales are in samples
```

Unknown metrics or parameter names are rejected at load time, and the settings used are
printed in the report.

Every metric also takes `min_samples` (off by default): recordings shorter than it get NaN for
that metric, and the report says how many and why, instead of quoting an unreliable value from a
short series. Sample entropy, for example, is usually given at least 10^m to 20^m samples
(Richman & Moorman, 2000), so `sample_entropy: {min_samples: 100}` for the default m = 2.
`dominant_period` and `dominant_power_share` also take `detrend: linear`, which removes a
trend that would otherwise put power at the longest periods.

With `axis: time`, a scale may be a duration instead (`30D`, `12h`, `15min`, `2W`, `10s`).
Each recording converts it to samples at its own sampling interval, and the report prints
both (`min_scale=30D = 30 samples at 1D`), so one profile means the same thing on daily and
hourly data. A whole number still means samples. Durations need a time column of dates or
timestamps, and they're rejected on the default `axis: samples` (which suits heart-rate
data, where scales count beats). Under `axis: time`, irregularly sampled recordings get no
metrics, and the report gives the reason, because every metric here assumes an even time grid.
A profile can also declare value units (`units: {tmax_anomaly: degC}`). These are printed in the
report and added to the signal-level rows of `condition_summary.csv` and
`significance_summary.csv`. Units aren't converted: the complexity metrics don't change
under a linear change of units, so the declared unit labels only the mean and the other descriptive statistics.

```yaml
axis: time
metric_parameters:
  dfa_exponent: {min_scale: 30D, max_scale: 3000D, order: 2}
units:
  tmax_anomaly: degC
```

A profile can derive signals before any metric runs, with `transforms`. Each transform makes
a signal from a column through steps applied within each recording, in time order: `diff`,
`log_return`, `abs`, `zscore`, `bandpass(low_hz, high_hz)` (zero-phase Butterworth) and
`envelope` (Hilbert amplitude). Selecting the source column analyzes the derived signals in
its place, and the report lists every transform applied. `bandpass` needs the sampling rate,
taken from a date/time column or from the profile's `sampling_interval`, since the engine
doesn't guess the unit of a plain-number time column.

```yaml
transforms:
  returns: {from: price, steps: [log_return]}
  volatility: {from: price, steps: [log_return, abs]}
  eeg_alpha: {from: eeg, steps: ["bandpass(8, 13)", envelope]}
sampling_interval: 0.00625s   # 160 Hz
```

A profile can ask for surrogate baselines, so a metric is never quoted without its null:
`surrogates: {type: shuffle, count: 200, seed: 0}` (defaults shown). Each metric is recomputed
on `count` surrogates of every recording. With `shuffle`, the copies keep the same values in
random order, so the test asks whether there is any temporal structure. With `phase`, the copies keep the power
spectrum and linear autocorrelation but nothing else (Theiler et al., 1992), so the test asks
whether there is structure beyond linear correlation. `complexity_by_unit.csv` gains
`<metric>_surrogate_mean` and a two-sided empirical `<metric>_surrogate_p`, and the report adds
a "Surrogate baselines" section. Surrogates are off unless a profile sets them, since they multiply
the metric computation time by `count`.

## Network pipeline: what it does today

For data that's naturally shaped as a graph (nodes and edges) rather than parallel time
series — e.g. a functional-connectivity network — `run_network_analysis.py` runs a separate,
parallel pipeline:

1. **Ingest** one or more CSV/Excel edge-list files (`source`, `target` columns, an optional
   `weight` column, and an optional `condition` column to compare multiple networks — e.g.
   `rest` vs `task`). Multiple files are merged if they share the same columns. Undirected by
   default; pass `--directed` to treat edges as directed. An optional node table
   (`--nodes nodes.csv`, with the id column set by `--node-col`, default `node`) lists nodes
   and their attributes. Listed nodes with no ties are added as isolates, so they count
   toward `n_nodes`, `density` and `mean_degree`. Attributes are joined onto
   `node_communities.csv` and `node_centrality.csv`. Edge endpoints missing from the table are kept,
   without attributes. A `condition` column in the table gives each condition its own nodes;
   without one, the table applies to every condition. The report's "Nodes" section counts
   each case. A node listed twice is an error.
   With a node table, `--attributes leaning,age` (or a profile's `node_attributes`) measures
   attribute mixing (homophily) per condition into `attribute_mixing.csv`. It reports Newman's
   assortativity, the share of ties within a group against the share random mixing would give,
   a permutation p-value (`--permutations`, default 1,000), and the agreement between the
   attribute and the detected communities (NMI). Numeric columns with more than two values
   are compared as numbers; anything else, including 0/1 codes, as groups, unless the profile
   sets `kind`.
2. **Clean**: drops rows missing a required value (source/target, and condition when used),
   and detects — but does not automatically remove — exact duplicate rows, mirroring the
   time-series pipeline's approach. Every input/dropped/output row count is recorded in a
   "Data provenance" report section.
3. **Compute network metrics** per condition (or once, for a single network): density, mean
   degree, average clustering, transitivity, modularity, average shortest path length, and
   number of connected components. For a directed network, also reports mean in-/out-degree
   and strongly- vs. weakly-connected component counts — see the report's note on how
   direction changes path-length and connectivity semantics.
4. **Compute per-node centrality** (degree, betweenness, eigenvector), opt-in via
   `--compute-centrality` since it's a different granularity (per-node, not per-condition)
   from the rest of the report. `--compute-communities` similarly writes each node's detected
   community (the partition modularity scores) to `node_communities.csv`, so it can be
   compared with known groups.
5. **Compute formal small-world statistics** (sigma and omega — Humphries & Gurney 2008;
   Telesford et al. 2011), opt-in via `--compute-small-world`, comparing the network's
   clustering/path length against randomized reference graphs of the same degree sequence
   rather than leaving "is this small-world?" to eyeballing `avg_clustering` and
   `avg_shortest_path_length`. Only well-defined for an undirected, fully connected network
   with at least 4 nodes — NaN otherwise. Off by default since it's the most computationally
   expensive metric in either pipeline.
6. **Visualize** the degree distribution per condition as a small-multiple bar chart.
7. **Interpret domain-aware**, grounded in the network-science literature already in the
   bibliography: small-world structure (Watts & Strogatz, 1998), scale-free/hub structure
   (Barabási & Albert, 1999), and community/modularity structure (Newman, 2010).
8. **Report**: writes `network_metrics_summary.csv`, optionally `node_centrality.csv` and
   `node_communities.csv`, the
   degree-distribution PNG, a `report.md`, and a self-contained `report.html`.

This is a genuinely different data shape from the time-series pipeline (rows are edges, not
observations), so it's kept as its own entry point (`network_analysis.py` /
`network_metrics.py`) rather than folded into `analyze_dataset()`. It reuses the same
`DomainProfile` configuration layer — see the `network` and `social_network` profiles in
`config/domain_profiles.yaml`.

## Behavior profiles (video and microscope tracking)

Four profiles interpret data exported by
[behavetrack](https://github.com/PJAllen5/behavior-analyzer), which tracks animals, people,
cells and neurons in video and microscope files. `behavetrack export-complexity` writes files
in exactly the shapes these profiles expect, and can run them directly.

| Profile | Pipeline | Data | Grounded in |
|---|---|---|---|
| `collective_behavior` | time series | per-frame group alignment (`polarization`), circling (`rotation`), speed and spacing; `session` = video | Vicsek et al., 1995; Couzin et al., 2002; Tunstrøm et al., 2013; Cavagna et al., 2010 |
| `neural_activity` | time series | ΔF/F per neuron per frame (`dff`), units `session,neuron_id`; DFA windows start at 16 frames, which still includes the indicator's smoothing (see the neural activity validation) | Chen et al., 2013; Linkenkaer-Hansen et al., 2001; Beggs & Plenz, 2003 |
| `animal_social_network` | network | who came near whom, weighted by seconds in contact | Krause et al., 2007; Wey et al., 2008; Farine & Whitehead, 2015; Lusseau & Newman, 2004 |
| `neuronal_network` | network | neuron pairs whose activity is correlated above a threshold | Bullmore & Sporns, 2009; Rubinov & Sporns, 2010; van Wijk et al., 2010; Bonifazi et al., 2009 |

```bash
complexity-analysis --data collective_timeseries.csv --domain collective_behavior \
  --signal-cols polarization,rotation,mean_speed_bl_s --unit-cols session
complexity-analysis --data neural_timeseries.csv --domain neural_activity \
  --signal-cols dff --unit-cols session,neuron_id
complexity-network --data contact_network.csv --domain animal_social_network
complexity-network --data functional_network.csv --domain neuronal_network
```

Their interpretation text points out what video tracking adds to the usual cautions: identity
swaps split one animal into several nodes and add noise that raises entropy; proximity
networks fill in with crowding and recording length, so compare against permuted networks;
calcium indicators smooth spiking, which lowers entropy and sets the shortest meaningful DFA
window; and correlation networks depend on the threshold, so compare them at equal density.

## License

MIT. See [LICENSE](LICENSE).

## Repository structure

```text
complexity-engine/
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml                   # makes it installable: pip install -e .
├── run_analysis.py                  # time-series CLI entry point (same as complexity-analysis)
├── run_network_analysis.py          # network CLI entry point (same as complexity-network)
├── requirements.txt
├── requirements-dev.txt             # + validation-only deps (antropy, nolds, scikit-learn)
├── complexity_engine/
│   ├── analysis.py                  # time-series ingestion, stats, report generation
│   ├── cli.py                       # command-line options for both pipelines
│   ├── domain_profiles.py           # domain profile registry + YAML/JSON loading (shared)
│   ├── metrics.py                   # sample entropy, spectral entropy, DFA exponent, mutual information, condition comparison
│   ├── network_analysis.py          # network ingestion, report generation
│   ├── network_metrics.py           # density, clustering, modularity, path length, ...
│   ├── plots.py                     # all plot generation, both pipelines
│   └── report_export.py             # self-contained HTML export, both pipelines
├── config/
│   ├── domain_profiles.yaml         # user-facing domain definitions, both pipelines (packaged into installs)
│   └── domain_profiles.json
├── data/
│   ├── physio_example.csv           # synthetic sample dataset for the time-series pipeline
│   ├── make_physio_example.py       # seeded generator for physio_example.csv
│   ├── fetch_fantasia.py            # downloads real heart-rate data for validation (not committed)
│   ├── fetch_ghcn_temperature.py    # downloads real daily temperature data for validation (not committed)
│   ├── fetch_social_networks.py     # writes real social networks for validation (not committed)
│   └── network_example.csv          # sample edge list for the network pipeline
├── outputs/                         # generated CSV summaries, plots/, and report.md
├── docs/
│   └── validation/
│       ├── fantasia.md              # known-answer validation on real heart-rate data
│       ├── gap_handling.md          # gap_policy validated on synthetic data loss and GHCN
│       ├── ghcn_temperature.md      # known-answer validation on real climate data
│       └── social_networks.md       # known-answer validation on real social networks
├── tests/                           # pytest suite
└── papers/
    └── references.bib               # curated complexity science / information theory / cognitive science bibliography
```

## Installing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # from a checkout; add ".[dev]" for tests and validation
# or, without cloning:
pip install "complexity-engine @ git+https://github.com/PJAllen5/complexity-engine.git"
```

This installs the package plus two commands, `complexity-analysis` (time series) and
`complexity-network`, with the same options as `run_analysis.py` and
`run_network_analysis.py`. An installed copy carries its domain profiles with it. Outside a
checkout, `--data` is required and results go to `./outputs`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .

python run_analysis.py \
  --data data/physio_example.csv \
  --signal-cols signal \
  --condition-col condition \
  --domain attention \
  --domain-config config/domain_profiles.yaml \
  --output-dir outputs

python run_network_analysis.py \
  --data data/network_example.csv \
  --output-dir outputs/network

# --data also accepts a comma-separated list of files (e.g. one per session) to merge:
python run_analysis.py --data session1.csv,session2.csv --signal-cols signal

# Recording units are auto-detected; override them, or pass `none` to treat each
# condition as a single recording:
python run_analysis.py --unit-cols subject_id,trial
```

Run the test suite with:

```bash
python -m pytest -q
```

A subset of tests (`tests/test_metric_validation.py`) cross-checks the complexity
metrics against independent reference implementations (`antropy`, `nolds`,
`scikit-learn`) and known theoretical values; they're skipped automatically unless
you install the validation-only dependencies:

```bash
pip install -r requirements-dev.txt
```

## Validation on real data

Checking that the metrics compute the right numbers isn't the same as checking that the
pipeline reaches the right conclusions on real data. `tests/test_fantasia_validation.py`
runs the whole pipeline on the PhysioNet Fantasia database (20 young and 20 elderly
healthy subjects, 120 minutes of resting heart rate each) and checks the age effects
published for that data. It's skipped unless you download the data first:

```bash
python data/fetch_fantasia.py   # ~0.7 MB of beat annotations -> data/external/fantasia_rr.csv
python -m pytest -q tests/test_fantasia_validation.py
```

The pipeline reproduces lower sample entropy and a higher short-range DFA exponent in the
elderly. It does not reproduce the published drop in the long-range DFA exponent, which
turned out to depend on the window it's fitted over. See
[docs/validation/fantasia.md](docs/validation/fantasia.md) for the full results.

`tests/test_gait_validation.py` checks the `gait` profile on PhysioNet stride intervals against
Hausdorff et al. (1997):

```bash
python data/fetch_gait.py   # ~2 MB of stride files -> data/external/gait_*.csv
python -m pytest -q tests/test_gait_validation.py
```

The drop in the stride-interval DFA exponent in Huntington's disease is reproduced (0.90 vs
0.67, p < 0.001), as is its link to functional capacity (ρ = 0.73). Aging goes in the published
direction (young 0.81, elderly 0.75), but with 5 subjects per group it's not significant. See
[docs/validation/gait.md](docs/validation/gait.md).

`tests/test_collective_behavior_synthetic.py` checks the `collective_behavior` profile on the
Vicsek et al. (1995) flocking model across noise levels. Polarization goes from 0.98 (ordered) to
0.11 (disordered), and its DFA exponent from about 1 to 0.55. The exponent doesn't clearly peak at
the transition, though the size of the fluctuations does. See
[docs/validation/collective_behavior_synthetic.md](docs/validation/collective_behavior_synthetic.md).

`tests/test_ghcn_validation.py` does the same outside biology, on daily maximum temperatures
from 18 GHCN-Daily stations, with the `climate_temperature` profile:

```bash
python data/fetch_ghcn_temperature.py   # ~17 MB of station records -> data/external/ghcn_tmax_anomaly.csv
python -m pytest -q tests/test_ghcn_validation.py
```

The pipeline reproduces long-range persistence near the published α ≈ 0.65 at every
continental station, with a narrow spread, and higher, more varied exponents on islands.
With only 5 island stations, the island difference is not significant (p = 0.075). See
[docs/validation/ghcn_temperature.md](docs/validation/ghcn_temperature.md).

`tests/test_lynx_validation.py` checks the `population_dynamics` profile on the Canadian lynx
trappings, 1821–1934:

```bash
python data/fetch_population_series.py   # 114 annual values -> data/external/lynx.csv
python -m pytest -q tests/test_lynx_validation.py
```

The dominant period is 9.5 years, matching the ten-year cycle of Elton & Nicholson (1942). That cycle
carries 52% of the variance, against 8% in shuffled copies (p = 0.005). See
[docs/validation/lynx.md](docs/validation/lynx.md).

`tests/test_neural_activity_synthetic.py` checks the `neural_activity` profile on simulated
calcium imaging, where the answer is known by construction. Correlated firing stands out from its
shuffles, and calcium smoothing lowers sample entropy as expected. But the indicator's smoothing
inflates the DFA exponent far beyond its decay time: uncorrelated spiking reads 0.69 from the
profile's 16-frame windows. See
[docs/validation/neural_activity_synthetic.md](docs/validation/neural_activity_synthetic.md).

`tests/test_social_network_validation.py` checks the network pipeline, with the
`social_network` profile, on Zachary's karate club and the ca-GrQc co-authorship network:

```bash
python data/fetch_social_networks.py   # karate club from networkx, ~0.1 MB from SNAP -> data/external/
python -m pytest -q tests/test_social_network_validation.py
```

ca-GrQc matches SNAP's published size, clustering, and giant component exactly, and is a
small world as Newman (2001) found for collaboration networks. On the karate club, modularity
matches (0.381), but the detected communities misassign 2 members against the known factions,
one more than the published methods. See
[docs/validation/social_networks.md](docs/validation/social_networks.md).

Attribute mixing is checked on the political blogs network (same fetch script as below):
91.15% of links stay on the linking blog's own side, against 91% published by Adamic & Glance
(2005), with assortativity 0.82 (p = 0.001). See
[docs/validation/attribute_mixing.md](docs/validation/attribute_mixing.md).

`tests/test_reference_network_validation.py` checks the `neuronal_network` and `network`
profiles on the two small-world networks of Watts & Strogatz (1998), the *C. elegans* neural
network and the US western power grid:

```bash
python data/fetch_reference_networks.py   # ~0.2 MB from networks.skewed.de -> data/external/
python -m pytest -q tests/test_reference_network_validation.py
```

The power grid matches the published node and edge counts exactly, with C = 0.080 and
L = 18.99 against 18.7. *C. elegans* (Newman's 297-neuron version) gives C = 0.29 and L = 2.46
against 0.28 and 2.65, with small-world sigma = 1.53. One pre-set criterion fails: it compared
the grid's path length with ln n / ln k, an estimate that's far too low for such a sparse
network. See [docs/validation/reference_networks.md](docs/validation/reference_networks.md).

`tests/test_dolphin_validation.py` checks the `animal_social_network` profile on Lusseau's
Doubtful Sound dolphins (same fetch script). Size, degree assortativity (−0.044) and the main
broker (SN100) match Lusseau & Newman (2004). Modularity is 0.496, just inside the 0.52 ± 0.03
of Newman & Girvan (2004). See [docs/validation/dolphins.md](docs/validation/dolphins.md).

## Literature base

`papers/references.bib` contains a curated bibliography covering:

- information theory and entropy
- cybernetics and systems theory
- complexity science and emergence
- network science, self-organization, and adaptation
- cognitive science, neuroscience, and Bayesian/information-based cognition

This is the theoretical grounding for the domain profiles and interpretation text used by
both pipelines above. Every inline citation in that text (e.g. "(Friston, 2010)") is checked
against this file by `tests/test_citation_integrity.py`, so a citation can't silently drift
out of sync with the bibliography (wrong year, removed entry, typo'd author).

## Design principles

- **Narrow and reproducible over broad and abstract.** Each pipeline targets one data shape
  with a condition/group comparison rather than trying to be a general-purpose stats package.
- **Domain-aware, not domain-hardcoded.** Adding or refining a scientific domain (its expected
  columns, relevant metrics, and interpretation language) is a config change, not a code change.
- **Explain, don't just compute.** Every metric in a report is paired with what it means and
  why it's relevant to that domain's guiding question.
- **A different data shape gets its own pipeline, not a branch.** The time-series and network
  pipelines share the domain-profile configuration layer but stay otherwise separate, rather
  than growing one function that branches on input shape.

## Suggested next steps

Tracked as GitHub issues rather than duplicated here — see the repo's issue tracker for the
full history (closed "Stage N" issues document what's shipped) and open issues for what's
next: unit-handling design (deliberately deferred until a real messy dataset motivates a
concrete approach).
