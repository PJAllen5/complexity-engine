# Complexity Analysis Report

- Domain: attention
- Domain description: Cognitive attention profile for task-driven dynamics and adaptive control.
- Data source: physio_example.csv
- Signal columns: signal
- Condition column: condition

## Condition summary
condition  mean_signal  std_signal  min_signal  max_signal  median signal
     rest    73.736667    1.192356        71.3        75.8   73.85 signal
     task    82.796667    2.994879        77.9        88.3   83.15 signal

## Complexity metrics
condition signal  n_samples      mean      std  min  max  variance  sample_entropy  spectral_entropy  dfa_exponent
     rest signal         30 73.736667 1.192356 71.3 75.8  1.421713        0.000000          0.859540      0.625685
     task signal         30 82.796667 2.994879 77.9 88.3  8.969299        0.980829          0.812203     -0.166528

## Interpretation
- Sample entropy captures temporal irregularity; higher values suggest more unpredictability.
- Spectral entropy captures the distribution of frequency content; higher values suggest broader frequency activity.
- DFA exponent estimates long-range correlation structure; values near 0.5 often indicate random-like structure, while higher values suggest stronger persistence.

- For signal, sample entropy is higher in task than in rest, suggesting a change in temporal irregularity or predictability.
### Domain interpretation rules
- sample_entropy: Higher entropy can indicate flexible attention allocation and less predictable task dynamics.
- spectral_entropy: Higher spectral entropy may indicate broader engagement or more distributed processing.
- dfa_exponent: Changes in DFA exponent can signal different forms of temporal coordination during attentional demands.