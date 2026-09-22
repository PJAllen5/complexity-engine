# Claude Handoff Brief

## Project
This repository is a small Python-based scientific analysis project for time-series complexity metrics, with a domain-aware configuration layer for scientific interpretation.

## Mission
Build a reproducible, small-scale system for analyzing time-series data using complexity metrics and domain-specific interpretation. The current product is intentionally narrow and domain-aware instead of broad or abstract.

## Current state
The project is implemented and working with a validated domain configuration layer.

### Core files
- run_analysis.py: CLI entry point
- complexity_engine/analysis.py: dataset loading, metric computation, summary generation, and report output
- complexity_engine/metrics.py: statistical and complexity metrics
- complexity_engine/domain_profiles.py: domain profile registry, loading logic, and configuration model
- config/domain_profiles.yaml: user-facing YAML definitions for domains like physiology, cognitive, stress, and attention
- tests/test_domain_profiles.py: regression tests for domain configuration and loading

## Verified working commands
Run from the project root:

```bash
. .venv/bin/activate && python -m pytest -q
. .venv/bin/activate && python run_analysis.py --data data/physio_example.csv --signal-cols signal --condition-col condition --domain attention --domain-config config/domain_profiles.yaml --output-dir outputs
```

## Current status
- 9 tests are passing
- YAML/JSON domain profiles load successfully
- CLI accepts a domain and optional custom config file
- Output report includes domain description and domain-specific interpretation rules

## Design direction
Keep the scope small and reproducible. The architecture is intentionally layered:
1. data ingestion
2. preprocessing / validation
3. descriptive statistics
4. condition comparison
5. complexity metrics
6. domain-aware interpretation
7. report output

Do not broaden too quickly. The next best refinements are:
- more granular domain definitions
- richer interpretation templates
- domain-specific question scaffolding
- richer multi-signal handling and report generation

## Useful context for continuation
- We chose Python because it matches the scientific ecosystem and complexity-analysis literature well.
- The project was intentionally constrained to a narrow first target: a physiological/cognitive time-series analysis pipeline.
- The domain configuration layer is now externalized so it can evolve without changing Python code paths.

## Recommended next step
Continue by extending the domain layer with more explicit scientific question templates and richer signal-role guidance before broadening to more general complexity analysis workflows.
