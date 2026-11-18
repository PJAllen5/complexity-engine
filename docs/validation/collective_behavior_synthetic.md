# Synthetic validation of the collective_behavior profile: the Vicsek model

Roadmap item A8, synthetic part ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
Real fish-school trajectories, and behavetrack end to end (A9), come later.

## Setup

The Vicsek et al. (1995) model has 100 self-propelled particles in a periodic 7 × 7 box (density
about 2), with speed 0.03 and interaction radius 1. Each step, a particle takes the mean heading of
its neighbours plus uniform noise in [−η/2, η/2]. Polarization (the length of the mean heading vector)
is recorded for 6,000 steps after a 1,000-step burn-in, with 2 runs per noise level. Each run is one
`session` in the pipeline, which uses the `collective_behavior` profile.

## Criteria (fixed before running) and results

| η | mean polarization | SD of polarization | DFA α (2 runs) | sample entropy |
|---|---|---|---|---|
| 0.5 | 0.983 | 0.005 | 1.02, 0.99 | 1.16, 1.25 |
| 1.5 | 0.862 | 0.037 | 1.01, 0.97 | 1.13, 1.38 |
| 2.5 | 0.624 | 0.095 | 0.99, 1.07 | 1.31, 1.23 |
| 3.5 | 0.312 | 0.117 | 0.90, 0.90 | 1.51, 1.51 |
| 5.0 | 0.106 | 0.054 | 0.54, 0.57 | 2.11, 2.10 |

1. **Order at low noise, disorder at high noise:** 0.983 (> 0.9) and 0.106 (< 0.25). ✓
2. **Polarization falls as noise rises:** at every step. ✓
3. **Fluctuations slowest near the transition:** the highest mean α is at η = 2.5 (1.03). ✓,
   **but not robustly.** It beats the ordered end (1.005 at η = 0.5) by only 0.025, less than the spread
   between replicate runs. α is about 1 throughout the ordered and transitional range and drops to about
   0.55 only in the disordered phase.

## What this shows

- The engine reads the model's order–disorder transition correctly: polarization, its persistence
  (α ≈ 1 when ordered, ≈ 0.55 when disordered), and its irregularity (sample entropy rising with noise).
- The size of the fluctuations (SD) peaks near the transition, as expected near a phase transition.
  Their persistence (DFA) doesn't single out the transition at this system size. So the profile's
  suggestion that a high exponent signals closeness to a transition should be read with care: an
  ordered group gives a similar exponent.

## References

- Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I. & Shochet, O. (1995). Novel type of phase transition in a system of self-driven particles. *Phys. Rev. Lett.* 75:1226–1229.
