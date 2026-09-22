# Complexity Analysis Report
- Domain: attention
- Domain description: Cognitive attention profile for task-driven dynamics and adaptive control.
- Analysis question: Does the monitored signal show greater organization or flexibility when attention is engaged by the task?
- Data source: physio_example.csv
- Signal columns: signal
- Condition column: condition
## Domain signal roles
- primary_signal: Main cognitive or neural signal associated with attentional state.
- condition: Task phase, condition, or experimental state used for comparison.
- covariate: Optional trial-level, session-level, or participant metadata.
Note: the 'attention' domain profile expects columns [signal, condition, trial]; missing from this dataset: trial.
## Condition summary
condition  mean_signal  std_signal  min_signal  max_signal  median signal
     rest    73.736667    1.192356        71.3        75.8   73.85 signal
     task    82.796667    2.994879        77.9        88.3   83.15 signal
## Complexity metrics
condition signal  n_samples      mean      std  min  max  variance  sample_entropy  spectral_entropy  dfa_exponent
     rest signal         30 73.736667 1.192356 71.3 75.8  1.421713        0.000000          0.859540      0.625685
     task signal         30 82.796667 2.994879 77.9 88.3  8.969299        0.980829          0.812203     -0.166528
## Interpretation
Guiding question: Does the monitored signal show greater organization or flexibility when attention is engaged by the task?
- sample_entropy: Higher entropy can indicate flexible attention allocation and less predictable task dynamics.
- spectral_entropy: Higher spectral entropy may indicate broader engagement or more distributed processing.
- dfa_exponent: Changes in DFA exponent can signal different forms of temporal coordination during attentional demands.
- For signal, sample entropy is higher in task than in rest. Higher entropy can indicate flexible attention allocation and less predictable task dynamics.
- For signal, spectral entropy is lower in task than in rest. Higher spectral entropy may indicate broader engagement or more distributed processing.
- For signal, dfa exponent is lower in task than in rest. Changes in DFA exponent can signal different forms of temporal coordination during attentional demands.