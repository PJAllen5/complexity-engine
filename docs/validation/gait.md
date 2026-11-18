# Validation on stride intervals: aging and Huntington's disease

Roadmap item B4 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This page covers the new `gait` profile, the first domain added from the roadmap, checked against Hausdorff et al. (1997).

## Data

`data/fetch_gait.py` downloads two open PhysioNet databases (Goldberger et al., 2000) from the
lab that published the result, and writes them to `data/external/` (not committed):

- **Gait in Aging and Disease** (`gaitdb`): 5 healthy young adults (21–29), 5 healthy elderly
  adults (71–77) and 5 people with Parkinson's disease. The healthy subjects walked for 15 minutes,
  giving 708–883 strides each after cleaning.
- **Gait in Neurodegenerative Disease** (`gaitndd`): 16 controls, 20 Huntington's disease (HD), 15
  Parkinson's, 13 ALS, each walking about 5 minutes (119–308 strides). The left foot's stride
  interval is used. Each HD subject's total functional capacity (TFC, where higher means less
  impaired) comes from `subject-description.txt`; one HD subject has no score.

These may not be exactly the paper's subjects. Strides more than 3 SD from each subject's mean
(turns, stumbles, missed strikes) were removed before analysis: 65 in `gaitdb` and 226 in `gaitndd`.

## Profile and criteria (fixed before the run)

The `gait` profile computes DFA1 on stride intervals from 16 strides up to a quarter of the series,
with scales counted in strides (`axis: samples`), as heart-rate scales count beats.

Published (Hausdorff et al., 1997, abstract): elderly 0.68 ± 0.14 vs young 0.87 ± 0.15
(P < 0.003); HD 0.60 ± 0.24 vs controls 0.88 ± 0.17 (P < 0.005); within HD, the exponent
correlates with functional capacity (r = 0.78). The criteria are in `tests/test_gait_validation.py`.

## Results

| comparison | group means (published) | p (Welch, one value per subject) | Cohen's d | criterion |
|---|---|---|---|---|
| young vs elderly (5 vs 5) | 0.807 (0.87) vs 0.751 (0.68) | 0.50 | −0.45 | direction ✓, young mean ✓, p ✗ |
| controls vs HD (16 vs 20) | 0.903 (0.88) vs 0.667 (0.60) | < 0.001 | 1.44 | ✓ |
| HD exponent vs TFC (n = 19) | Spearman ρ = 0.73 (Pearson r = 0.69; published 0.78) | 0.0004 | | ✓ |

- **Huntington's disease: reproduced.** The size, direction and significance of the drop all
  match, and so does its link to functional impairment.
- **Aging: partial (strict xfail).** Young adults show the published long-range correlations, and the
  elderly mean is lower, but with 5 subjects per group the difference isn't close to significant.
  One young subject (y3, α = 0.654) sits below most elderly subjects. The young group's spread
  (SD 0.15) matches the paper, so the gap is mostly a power problem: at the paper's effect size
  (d ≈ 1.3), about 10 subjects per group are needed for 80% power. The elderly mean (0.751) is
  also higher than published (0.68).
- Not criteria, but reported: ALS (0.792) and Parkinson's (0.788 in `gaitndd`, 0.743 in
  `gaitdb`) fall between controls and HD, and none differs from controls at p < 0.05.

## References

- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101:e215–e220.
- Hausdorff, J. M., Peng, C.-K., Ladin, Z., Wei, J. Y. & Goldberger, A. L. (1995). Is walking a random walk? *J Appl Physiol* 78:349–358.
- Hausdorff, J. M., Mitchell, S. L., Firtion, R., Peng, C.-K., Cudkowicz, M. E., Wei, J. Y. & Goldberger, A. L. (1997). Altered fractal dynamics of gait: reduced stride-interval correlations with aging and Huntington's disease. *J Appl Physiol* 82:262–269.
