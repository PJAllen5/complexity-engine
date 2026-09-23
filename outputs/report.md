# Complexity Analysis Report

- Domain: attention
- Domain description: Cognitive attention profile for task-driven dynamics and adaptive control.
- Analysis question: Does the monitored signal show tighter predictive precision when attention is engaged by the task, or reduced top-down control?
- Data source: physio_example.csv
- Signal columns: signal
- Condition column: condition
- Recording unit columns: subject_id

## Domain signal roles
- primary_signal: Main cognitive or neural signal associated with attentional state.
- condition: Task phase, condition, or experimental state used for comparison.
- covariate: Optional trial-level, session-level, or participant metadata.

Note: the 'attention' domain profile expects columns [signal, condition, trial]; missing from this dataset: trial.

## Data provenance
- Input rows: 1800
- Duplicate rows detected (not removed automatically): 0
- Rows dropped for missing values in ['signal', 'condition']: 0
- Rows analyzed: 1800

## Condition summary
```
condition  mean_signal  std_signal  min_signal  max_signal  median signal
     rest    72.254667    3.310537        64.2        81.2    72.3 signal
     task    84.331333    2.116774        78.8        90.1    84.3 signal
```

## Complexity metrics
Complexity metrics are computed separately within each recording unit (subject_id; 300 samples per unit), sorted by timestamp, then averaged across units per condition (n_units). Per-unit values are in complexity_by_unit.csv.

```
condition signal  n_samples  n_units      mean      std  min  max  variance  sample_entropy  spectral_entropy  dfa_exponent
     rest signal        900        3 72.254667 3.310537 64.2 81.2 10.959656        1.607469          0.481321      1.215614
     task signal        900        3 84.331333 2.116774 78.8 90.1  4.480730        2.016142          0.816005      0.813603
```

## Statistical significance
```
signal condition_a condition_b  n_a  n_b    mean_a    mean_b  t_statistic  p_value  cohens_d
signal        rest        task  900  900 72.254667 84.331333   -92.201771      0.0 -4.346433
```

## Visualizations
![signal_condition_comparison](plots/signal_condition_comparison.png)
![signal_complexity_metrics](plots/signal_complexity_metrics.png)

## Interpretation

Guiding question: Does the monitored signal show tighter predictive precision when attention is engaged by the task, or reduced top-down control?

Methodological note: entropy-based metrics quantify unpredictability, not complexity in the fuller sense used in complexity science (Crutchfield & Young, 1989; Gell-Mann & Lloyd, 1996). A maximally random signal has high entropy but little structured organization, so read the entropy changes below as changes in signal regularity, not as a direct verdict on how "complex" the underlying system is.

Welch's t-test checks whether the raw signal's mean level differs between two conditions beyond what sampling variability alone would produce; Cohen's d expresses the size of that difference in standard-deviation units (roughly: ~0.2 small, ~0.5 medium, ~0.8 large). This tests the raw signal, not the complexity metrics above, which are single summary values per condition rather than per-sample distributions.

- sample_entropy: Attention can be framed as precision-weighting of expected sensory or task-relevant signals (Friston, 2010; Clark, 2013; Hohwy, 2013); reduced entropy when attention is engaged is consistent with tighter predictive precision around the task, while elevated entropy may reflect divided attention or reduced top-down control rather than richer processing.
- spectral_entropy: A shift toward broader frequency engagement when attention is recruited is consistent with multiple task-relevant processes being active concurrently (Rumelhart et al., 1986); a narrower spectrum can indicate a single dominant rhythm capturing the bulk of signal energy.
- dfa_exponent: Changes in long-range temporal correlation during attentional engagement point to shifts in how strongly past states constrain present ones (Hopfield, 1982); stronger persistence is consistent with sustained, attractor-like engagement, while a drop toward random-walk-like behavior (~0.5) suggests more moment-to-moment reactivity.

- sample entropy is higher in task than in rest. Attention can be framed as precision-weighting of expected sensory or task-relevant signals (Friston, 2010; Clark, 2013; Hohwy, 2013); reduced entropy when attention is engaged is consistent with tighter predictive precision around the task, while elevated entropy may reflect divided attention or reduced top-down control rather than richer processing.
- spectral entropy is higher in task than in rest. A shift toward broader frequency engagement when attention is recruited is consistent with multiple task-relevant processes being active concurrently (Rumelhart et al., 1986); a narrower spectrum can indicate a single dominant rhythm capturing the bulk of signal energy.
- dfa exponent is lower in task than in rest. Changes in long-range temporal correlation during attentional engagement point to shifts in how strongly past states constrain present ones (Hopfield, 1982); stronger persistence is consistent with sustained, attractor-like engagement, while a drop toward random-walk-like behavior (~0.5) suggests more moment-to-moment reactivity.
- Raw signal differs significantly between rest and task (Welch's t-test p<0.001, Cohen's d=-4.35).