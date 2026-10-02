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
   signals whose correlation structure changes with scale — restricted to whichever metrics
   the active domain profile recommends. These metrics depend on sample order, so they're computed
   separately within each independent recording (subject, trial, session — auto-detected from
   `subject_id`/`subject`/`participant`/`session`/`trial`, or set with `--unit-cols`), sorted by
   a `timestamp`/`time`/`timepoint` column when present, then averaged across recordings per
   condition. Joining recordings end to end would insert artificial jumps that the metrics
   would read as real dynamics. Per-recording values are written to `complexity_by_unit.csv`.
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
   `significance_summary.csv`, `cross_signal_summary.csv`, the plot PNGs, a human-readable
   `report.md`, and a self-contained `report.html` (same content, styled, with the plots
   base64-inlined into the single file) to an output directory.

Domain profiles for this pipeline (`physiology`, `cognitive`, `stress`, `attention`,
`heart_rate_variability`, and any
custom ones you add) live in `config/domain_profiles.yaml` (or `.json`) and can be extended or
overridden without touching Python code. A profile's `recommended_metrics` is validated
against the pipeline's known metrics as soon as the domain is looked up — before any data is
loaded — so a typo or a metric name that belongs to the other pipeline fails immediately with
a clear message, not deep inside a later analysis step.

## Network pipeline: what it does today

For data that's naturally shaped as a graph (nodes and edges) rather than parallel time
series — e.g. a functional-connectivity network — `run_network_analysis.py` runs a separate,
parallel pipeline:

1. **Ingest** one or more CSV/Excel edge-list files (`source`, `target` columns, an optional
   `weight` column, and an optional `condition` column to compare multiple networks — e.g.
   `rest` vs `task`). Multiple files are merged if they share the same columns. Undirected by
   default; pass `--directed` to treat edges as directed.
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
   from the rest of the report.
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
8. **Report**: writes `network_metrics_summary.csv`, optionally `node_centrality.csv`, the
   degree-distribution PNG, a `report.md`, and a self-contained `report.html`.

This is a genuinely different data shape from the time-series pipeline (rows are edges, not
observations), so it's kept as its own entry point (`network_analysis.py` /
`network_metrics.py`) rather than folded into `analyze_dataset()`. It reuses the same
`DomainProfile` configuration layer — see the `network` profile in
`config/domain_profiles.yaml`.

## Repository structure

```text
complexity-engine/
├── README.md
├── run_analysis.py                  # time-series CLI entry point
├── run_network_analysis.py          # network CLI entry point
├── requirements.txt
├── requirements-dev.txt             # + validation-only deps (antropy, nolds, scikit-learn)
├── complexity_engine/
│   ├── analysis.py                  # time-series ingestion, stats, report generation
│   ├── domain_profiles.py           # domain profile registry + YAML/JSON loading (shared)
│   ├── metrics.py                   # sample entropy, spectral entropy, DFA exponent, mutual information, condition comparison
│   ├── network_analysis.py          # network ingestion, report generation
│   ├── network_metrics.py           # density, clustering, modularity, path length, ...
│   ├── plots.py                     # all plot generation, both pipelines
│   └── report_export.py             # self-contained HTML export, both pipelines
├── config/
│   ├── domain_profiles.yaml         # user-facing domain definitions, both pipelines
│   └── domain_profiles.json
├── data/
│   ├── physio_example.csv           # synthetic sample dataset for the time-series pipeline
│   ├── make_physio_example.py       # seeded generator for physio_example.csv
│   ├── fetch_fantasia.py            # downloads real heart-rate data for validation (not committed)
│   └── network_example.csv          # sample edge list for the network pipeline
├── outputs/                         # generated CSV summaries, plots/, and report.md
├── docs/
│   └── validation/fantasia.md       # known-answer validation on real heart-rate data
├── tests/                           # pytest suite
└── papers/
    └── references.bib               # curated complexity science / information theory / cognitive science bibliography
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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
