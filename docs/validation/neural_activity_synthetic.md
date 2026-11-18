# Synthetic validation of the neural_activity profile

Roadmap item A7, synthetic part ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
The real-data part (Allen Brain Observatory) needs NWB conversion and comes later.

## Setup

Each simulated neuron is a spike train (Poisson, or with a rate that follows long-range-correlated
noise, fGn with H = 0.9), convolved with an exponential calcium transient and given imaging noise.
There are 10 neurons per condition and 6,000 frames each. This mirrors how ΔF/F reports spiking
through an indicator (Chen et al., 2013). The answers are known by construction, and the criteria in
`tests/test_neural_activity_synthetic.py` were fixed before running.

## Results

| check | criterion | result |
|---|---|---|
| (a) Poisson spiking, 10 fps, 0.5 s decay, profile settings | α within 0.5 ± 0.1 | **0.69** ✗ (strict xfail) |
| (b) correlated firing, same imaging | α > 0.7 | 0.81 ✓ |
| (b) vs (a) | at least 0.15 apart | 0.12 ✗ (strict xfail) |
| (b) vs shuffle surrogates | ≥ 80% of neurons p < 0.05 | ✓ |
| (c) calcium smoothing vs no indicator | lower sample entropy, every neuron | ✓ |
| (d) 30 fps, 1.5 s decay (45 frames), min_scale 16 | α > 0.6 | 1.05 ✓ |
| (d) same, min_scale 180 (4 decay times) | α within 0.5 ± 0.1 | **0.75** ✗ (strict xfail) |

## What failed, and what it means

The same spikes without the indicator give α = 0.51, as they should. Every failure has one cause:
**the indicator's smoothing inflates the DFA exponent far beyond its decay time.** The criteria
assumed the effect ends within 3–4 decay times. It doesn't:

| min_scale (in decay times, 45-frame decay) | α of Poisson spiking |
|---|---|
| 16 frames (0.4) | 1.05 |
| 45 (1) | 0.94 |
| 180 (4) | 0.75 |
| 400 (9) | 0.65 |

Smoothed Gaussian noise behaves the same way (5-frame decay: 0.67 from 16 frames, 0.56 from 50).
So it's a property of DFA on smoothed signals, not of spiking.

For the profile, this means:

- The previous comment said 16 frames "stays above the decay of a calcium transient". That was
  wrong, and the comment now says what the simulation shows. **The value itself is unchanged** (no
  output changes). Raising it is a decision for the project. About 10 decay times (e.g. 50 frames
  at 10 fps with GCaMP6f-like kinetics) brings uncorrelated spiking near 0.5, but it needs longer
  recordings: DFA uses windows up to a quarter of the series.
- Comparisons between conditions recorded with the same indicator and frame rate stay valid,
  because both are inflated alike. Correlated firing still separates from Poisson firing (0.81 vs
  0.69) and from its own shuffles. An absolute value near 0.7 from ΔF/F, however, isn't evidence of
  long-range correlation by itself. Surrogates help only partly, because shuffling also destroys the
  indicator's smoothing. A fairer null would be Poisson spikes through the same kernel, which is
  what case (a) is.

## References

- Chen, T.-W. et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. *Nature* 499:295–300.
- Peng, C.-K., Havlin, S., Stanley, H. E. & Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos* 5:82–87.
