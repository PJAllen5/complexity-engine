# Complexity Engine

A small, reproducible Python pipeline for domain-aware complexity analysis of scientific
time-series data. It ingests raw CSV/Excel data, computes standard descriptive statistics,
applies a domain-specific set of complexity metrics (entropy, spectral, and long-range
correlation measures), and generates a report that explains the results in the language of
the chosen scientific domain.

The project started as a literature review workspace for complexity science and information
theory (see `papers/references.bib`) and has since grown into a working analysis tool built on
that foundation.

## What it does today

1. **Ingest** a CSV or Excel file of time-series data (one or more signal columns, a
   condition/group column, and optional metadata like subject ID or trial).
2. **Summarize** each signal by condition (mean, std, min, max, median).
3. **Compute complexity metrics** per condition — sample entropy, spectral entropy, and a DFA
   (detrended fluctuation analysis) scaling exponent — restricted to whichever metrics the
   active domain profile recommends.
4. **Compute cross-signal coupling**: when more than one signal column is selected, pairwise
   mutual information is computed per condition, capturing how tightly signals are coupled.
5. **Interpret domain-aware**: each domain profile supplies a guiding analysis question,
   descriptions of signal roles (primary signal / condition / covariate), and
   metric-specific interpretation text (including for cross-signal coupling). The generated
   report uses these to explain *why* a higher or lower value matters for that domain, rather
   than just printing numbers. With multiple signals, the interpretation is grouped into a
   subsection per signal plus a cross-signal relationships subsection.
6. **Report**: writes `condition_summary.csv`, `complexity_summary.csv`,
   `cross_signal_summary.csv`, and a human-readable `report.md` to an output directory.

Domain profiles (`physiology`, `cognitive`, `stress`, `attention`, and any custom ones you
add) live in `config/domain_profiles.yaml` (or `.json`) and can be extended or overridden
without touching Python code.

## Repository structure

```text
complexity-engine/
├── README.md
├── run_analysis.py                  # CLI entry point
├── requirements.txt
├── complexity_engine/
│   ├── analysis.py                  # ingestion, stats, report generation
│   ├── domain_profiles.py           # domain profile registry + YAML/JSON loading
│   └── metrics.py                   # sample entropy, spectral entropy, DFA exponent, mutual information
├── config/
│   ├── domain_profiles.yaml         # user-facing domain definitions
│   └── domain_profiles.json
├── data/
│   └── physio_example.csv           # sample dataset for the physiology/attention domains
├── outputs/                         # generated CSV summaries + report.md
├── tests/                           # pytest suite for metrics, domains, and analysis
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

This is the theoretical grounding for the domain profiles and interpretation text used by the
analysis pipeline above.

## Design principles

- **Narrow and reproducible over broad and abstract.** The pipeline targets time-series data
  with a condition/group comparison rather than trying to be a general-purpose stats package.
- **Domain-aware, not domain-hardcoded.** Adding or refining a scientific domain (its expected
  columns, relevant metrics, and interpretation language) is a config change, not a code change.
- **Explain, don't just compute.** Every complexity metric in a report is paired with what it
  means and why it's relevant to that domain's guiding question.

## Suggested next steps

- surface missing-expected-column warnings programmatically, not just in the report text
- expand the domain profile library and cross-reference profiles with the bibliography
- explore network-based metrics (degree, clustering, modularity) for data shaped as a network
  rather than parallel time series
