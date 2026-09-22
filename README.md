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

1. **Ingest** a CSV or Excel file of time-series data (one or more signal columns, a
   condition/group column, and optional metadata like subject ID or trial).
2. **Summarize** each signal by condition (mean, std, min, max, median).
3. **Compute complexity metrics** per condition — sample entropy, spectral entropy, and a DFA
   (detrended fluctuation analysis) scaling exponent — restricted to whichever metrics the
   active domain profile recommends.
4. **Test statistical significance** between every pair of conditions on the raw signal
   (Welch's t-test + Cohen's d effect size), so directional differences can be read alongside
   whether they're distinguishable from sampling noise.
5. **Compute cross-signal coupling**: when more than one signal column is selected, pairwise
   mutual information is computed per condition, capturing how tightly signals are coupled.
6. **Interpret domain-aware**: each domain profile supplies a guiding analysis question,
   descriptions of signal roles (primary signal / condition / covariate), and
   metric-specific interpretation text (including for cross-signal coupling). The generated
   report uses these to explain *why* a higher or lower value matters for that domain, rather
   than just printing numbers. With multiple signals, the interpretation is grouped into a
   subsection per signal plus a cross-signal relationships subsection.
7. **Visualize**: generates a condition-comparison box plot and a complexity-metrics bar
   chart per signal, plus a mutual-information heatmap when multiple signals are selected —
   saved as PNGs under `outputs/plots/` and embedded in `report.md`.
8. **Report**: writes `condition_summary.csv`, `complexity_summary.csv`,
   `significance_summary.csv`, `cross_signal_summary.csv`, the plot PNGs, and a human-readable
   `report.md` to an output directory.

Domain profiles for this pipeline (`physiology`, `cognitive`, `stress`, `attention`, and any
custom ones you add) live in `config/domain_profiles.yaml` (or `.json`) and can be extended or
overridden without touching Python code.

## Network pipeline: what it does today

For data that's naturally shaped as a graph (nodes and edges) rather than parallel time
series — e.g. a functional-connectivity network — `run_network_analysis.py` runs a separate,
parallel pipeline:

1. **Ingest** a CSV/Excel edge list (`source`, `target` columns, an optional `weight` column,
   and an optional `condition` column to compare multiple networks — e.g. `rest` vs `task`).
2. **Compute network metrics** per condition (or once, for a single network): density, mean
   degree, average clustering, transitivity, modularity, average shortest path length, and
   number of connected components.
3. **Visualize** the degree distribution per condition as a small-multiple bar chart.
4. **Interpret domain-aware**, grounded in the network-science literature already in the
   bibliography: small-world structure (Watts & Strogatz, 1998), scale-free/hub structure
   (Barabási & Albert, 1999), and community/modularity structure (Newman, 2010).
5. **Report**: writes `network_metrics_summary.csv`, the degree-distribution PNG, and a
   `report.md`.

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
├── complexity_engine/
│   ├── analysis.py                  # time-series ingestion, stats, report generation
│   ├── domain_profiles.py           # domain profile registry + YAML/JSON loading (shared)
│   ├── metrics.py                   # sample entropy, spectral entropy, DFA exponent, mutual information, condition comparison
│   ├── network_analysis.py          # network ingestion, report generation
│   ├── network_metrics.py           # density, clustering, modularity, path length, ...
│   └── plots.py                     # all plot generation, both pipelines
├── config/
│   ├── domain_profiles.yaml         # user-facing domain definitions, both pipelines
│   └── domain_profiles.json
├── data/
│   ├── physio_example.csv           # sample dataset for the time-series pipeline
│   └── network_example.csv          # sample edge list for the network pipeline
├── outputs/                         # generated CSV summaries, plots/, and report.md
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
```

Run the test suite with:

```bash
python -m pytest -q
```

## Literature base

`papers/references.bib` contains a curated bibliography covering:

- information theory and entropy
- cybernetics and systems theory
- complexity science and emergence
- network science, self-organization, and adaptation
- cognitive science, neuroscience, and Bayesian/information-based cognition

This is the theoretical grounding for the domain profiles and interpretation text used by
both pipelines above.

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

- expand the domain profile library and cross-reference profiles with the bibliography
- for the network pipeline: directed-graph support, per-node centrality output, and a
  random-graph baseline comparison for a formal small-world "sigma" statistic
