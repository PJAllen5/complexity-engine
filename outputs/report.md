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

## Sampling
Each unit's sampling interval is the median spacing of 'timestamp'; a gap is a spacing over 1.5 intervals. Per-unit values are in sampling_by_unit.csv.

- Every unit is regularly sampled with no gaps.

```
condition subject_id  n_samples interval  coverage  n_gaps longest_gap  duplicate_times  spacing_cv  irregular
     rest        s01        300       1s       1.0       0                            0         0.0      False
     rest        s02        300       1s       1.0       0                            0         0.0      False
     rest        s03        300       1s       1.0       0                            0         0.0      False
     task        s01        300       1s       1.0       0                            0         0.0      False
     task        s02        300       1s       1.0       0                            0         0.0      False
     task        s03        300       1s       1.0       0                            0         0.0      False
```

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
     rest signal        900        3 72.254667 3.310537 64.2 81.2 10.959656        1.607469          0.481321      1.243296
     task signal        900        3 84.331333 2.116774 78.8 90.1  4.480730        2.016142          0.816005      0.787843
```

## Statistical significance
```
signal condition_a condition_b          measure                      test  n_a  n_b    mean_a    mean_b  t_statistic  p_value   cohens_d
signal        rest        task      signal_mean welch_on_recording_values  3.0  3.0 72.254667 84.331333    -5.645168 0.007016  -4.609261
signal        rest        task   sample_entropy welch_on_recording_values  3.0  3.0  1.607469  2.016142    -3.469878 0.028059  -2.833144
signal        rest        task spectral_entropy welch_on_recording_values  3.0  3.0  0.481321  0.816005   -18.236326 0.002422 -14.889897
signal        rest        task     dfa_exponent welch_on_recording_values  3.0  3.0  1.243296  0.787843     7.434849 0.001871   6.070529
```

## Visualizations
![signal_condition_comparison](plots/signal_condition_comparison.png)
![signal_complexity_metrics](plots/signal_complexity_metrics.png)

## Interpretation

Guiding question: Does the monitored signal show tighter predictive precision when attention is engaged by the task, or reduced top-down control?

Methodological note: entropy-based metrics quantify unpredictability, not complexity in the fuller sense used in complexity science (Crutchfield & Young, 1989; Gell-Mann & Lloyd, 1996). A maximally random signal has high entropy but little structured organization, so read the entropy changes below as changes in signal regularity, not as a direct verdict on how "complex" the underlying system is.

Welch's t-test checks whether two conditions differ beyond what sampling variability alone would produce; Cohen's d expresses the size of that difference in standard-deviation units (roughly: ~0.2 small, ~0.5 medium, ~0.8 large). Consecutive samples of a time series are not independent observations, and neither are samples from the same recording, so the raw samples are never tested as if they were. When each condition has at least two recordings, the test compares one value per recording (its mean signal level, and each complexity metric), so n is the number of recordings. Recordings of the same subjects in both conditions are treated as unpaired, which is conservative. When a condition has only one recording, the raw samples are tested with each recording's sample size reduced to the number of independent samples its lag-1 autocorrelation implies (Zwiers & von Storch, 1995); this correction is approximate and can still overstate significance for strongly persistent signals, so treat those p-values as a lower bound.

- sample_entropy: Attention can be framed as precision-weighting of expected sensory or task-relevant signals (Friston, 2010; Clark, 2013; Hohwy, 2013); reduced entropy when attention is engaged is consistent with tighter predictive precision around the task, while elevated entropy may reflect divided attention or reduced top-down control rather than richer processing.
- spectral_entropy: A shift toward broader frequency engagement when attention is recruited is consistent with multiple task-relevant processes being active concurrently (Rumelhart et al., 1986); a narrower spectrum can indicate a single dominant rhythm capturing the bulk of signal energy.
- dfa_exponent: Changes in long-range temporal correlation during attentional engagement point to shifts in how strongly past states constrain present ones (Hopfield, 1982); stronger persistence is consistent with sustained, attractor-like engagement, while a drop toward random-walk-like behavior (~0.5) suggests more moment-to-moment reactivity.

- sample entropy is higher in task than in rest. Attention can be framed as precision-weighting of expected sensory or task-relevant signals (Friston, 2010; Clark, 2013; Hohwy, 2013); reduced entropy when attention is engaged is consistent with tighter predictive precision around the task, while elevated entropy may reflect divided attention or reduced top-down control rather than richer processing.
- spectral entropy is higher in task than in rest. A shift toward broader frequency engagement when attention is recruited is consistent with multiple task-relevant processes being active concurrently (Rumelhart et al., 1986); a narrower spectrum can indicate a single dominant rhythm capturing the bulk of signal energy.
- dfa exponent is lower in task than in rest. Changes in long-range temporal correlation during attentional engagement point to shifts in how strongly past states constrain present ones (Hopfield, 1982); stronger persistence is consistent with sustained, attractor-like engagement, while a drop toward random-walk-like behavior (~0.5) suggests more moment-to-moment reactivity.
- Mean signal level differs significantly between rest and task (Welch's t-test on 3 vs 3 recordings, p=0.007, Cohen's d=-4.61).
- sample entropy differs significantly between rest and task (Welch's t-test on 3 vs 3 recordings, p=0.028, Cohen's d=-2.83).
- spectral entropy differs significantly between rest and task (Welch's t-test on 3 vs 3 recordings, p=0.002, Cohen's d=-14.89).
- dfa exponent differs significantly between rest and task (Welch's t-test on 3 vs 3 recordings, p=0.002, Cohen's d=6.07).