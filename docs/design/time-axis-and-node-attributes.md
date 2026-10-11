# Design: a real time axis for series, and node attributes for networks

Status: agreed 2026-10-10 for #59, with part of #35 (decisions at the end). Each part is driven by a real dataset that didn't
fit, as #59 asked. The climate run (#57, [ghcn_temperature.md](../validation/ghcn_temperature.md))
showed the time-axis gaps. The social-network run (#58, [social_networks.md](../validation/social_networks.md))
showed the missing node attributes.

## Framing: extend two shapes rather than add a third

#59 expected a new entry point per data shape. Neither gap needs one:

- A gappy daily temperature record is still "one ordered series per recording". What's
  missing is that the engine doesn't know *where* in time each sample sits.
- An attributed social network is still "an edge list". What's missing is a table describing
  the nodes.

So both are extensions of the existing pipelines, behind opt-in inputs and profile settings.
Each existing run keeps producing the same output unless it opts in. Panel/survey data, event
lists, and genuinely irregular sampling (cores, light curves) remain candidates for a real
third shape under #59.

---

## Part A — Time series: a time axis, gaps, and time units

### What went wrong in #57

1. **Gaps are invisible.** Rows are sorted by `time` and then treated as consecutive samples.
   A 3-day gap in a daily record becomes two adjacent days. The GHCN fetch script avoided this
   by dropping 5 of 23 stations below 80% coverage, including the most continental one, Chita.
2. **Scales are in samples.** `dfa_exponent: {min_scale: 30}` means 30 days only because
   there's one sample per day. The same profile on hourly data would silently mean 30 hours.
3. **`date` isn't a time column.** Only `timestamp`, `time` and `timepoint` are recognized.

The same problem already appears in Fantasia, without being reported. Excluding beats next to
ectopic beats leaves holes in the NN-interval sequence. f2o08 lost 18% of its intervals, and
the engine joins across them.

### Evidence that should set the default

Ma et al. (2010, *Phys Rev E* 81:031101) removed segments from long-range correlated signals
and joined the rest:

- **Positively correlated signals** (α > 0.5, as in heart rate and temperature): the DFA
  exponent is practically unchanged even with 90% of the data removed.
- **Anticorrelated signals** (α < 0.5): the exponent drifts toward 0.5 with as little as 10%
  removed.

So joining is a sound default for the domains validated so far, but not a safe general one.
The engine should report gaps always, join by default (no change to existing results), and
offer a segment-aware mode that is correct for any signal.

### Proposed behaviour

**1. Time axis and gap report (always on when a time column exists).** For each recording
unit, the engine reads the time column (numeric or datetime, with `date` added to the
recognized names). It then infers the nominal sampling interval as the median spacing,
unless the profile declares one (`sampling_interval: 1D`). A gap is any spacing greater than
1.5× the interval. The report gets a table per unit:

| unit | n_samples | interval | coverage | n_gaps | longest_gap | spacing_cv |
|---|---|---|---|---|---|---|

`spacing_cv` is the coefficient of variation of the spacings with the gaps removed. Above
0.1, the series is flagged as *irregularly sampled*: not gappy-but-regular, and outside what
the regular-grid metrics assume. RR intervals are irregular in clock time by nature, which is
why the next setting exists.

**2. Profile setting `axis: samples | time`** (default `samples`). HRV analyses count scales in
beats, not seconds, so `heart_rate_variability` stays on `samples`. With `axis: time`, the
irregular-sampling flag makes the time-based metrics NaN, with a reason in the report. They
are never computed as if the grid were regular.

**3. Profile setting `gap_policy: join | segment | exclude`** (default `join`, today's
behaviour, now reported):
- `join` — concatenate across gaps. Correct for positively correlated signals (Ma et al.,
  2010). The report states how many joins happened.
- `segment` — split each unit into contiguous segments at gaps. DFA builds each segment's
  profile separately and pools squared residuals only from windows that sit wholly inside one
  segment. Sample entropy counts only templates (m + 1 points) that don't cross a gap. Spectral
  entropy averages segment spectra (Welch-style) over segments at least as long as the FFT
  window. Segments too short for any window just don't contribute. A unit is NaN, with a
  reason, only if nothing is left.
- `exclude` — drop units below `min_coverage`. This is what the GHCN fetch script did, moved
  into the engine and reported.

**4. Time-unit scales.** With `axis: time`, `metric_parameters` scales may be durations:
`dfa_exponent: {min_scale: 30D, max_scale: 3000D, order: 2}`. They're converted to samples
using the nominal interval, and the report prints both (`30D = 30 samples at 1D`). A bare
integer still means samples, so existing profiles don't change. A duration on a profile with
no time column, or with `axis: samples`, is a load-time error.

**5. Value units (#35, scoped down).** Every complexity metric here is invariant to linear
unit changes: sample entropy uses r = 0.2 × SD, DFA is a log-log slope, and spectral entropy
is normalized. A mmHg/kPa mismatch therefore doesn't change those metrics. It only distorts
the pooled descriptive statistics and the `signal_mean` comparison. The proposal is that a
profile may declare `units: {tmax_anomaly: degC}`, the report prints it, and
`signal_mean`/descriptive rows carry the unit. Conversion stays deferred in #35 until a real
dataset with mismatched units turns up. We still haven't seen one; GHCN's tenths of a degree
was a fixed format, handled at fetch time.

### Validation

- **Synthetic, against a known answer.** Reproduce Ma et al. (2010) qualitatively. Generate
  fractional Gaussian noise with α = 0.3, 0.5, 0.7 and 0.9, and remove 10–50% in random
  segments. `join` should hold α for 0.7/0.9 and drift for 0.3. `segment` should hold α for
  all four within a tolerance fixed in advance.
- **Real.** Rerun GHCN with `gap_policy: segment` and no coverage exclusion. That adds Perm,
  Chita, Minsk, Brno and Hong Kong (18 continental vs 5 island stations), with settings and
  hypotheses fixed beforehand. Chita also gives a first real look at Fraedrich & Blender's
  inner-continental α ≈ 0.5 claim, which #57 couldn't test.
- **Regression.** Fantasia and the example outputs are unchanged under the default `join`.
  The new gap report shows the Fantasia joins that were invisible before.

---

## Part B — Networks: a node table

### What went wrong in #58

1. **No node attributes.** Homophily, segregation, and "do communities follow known groups?"
   are central to social network analysis and can't be expressed. The karate factions lived
   in a side file joined inside a test.
2. **Isolates are lost.** An edge list can't contain a node with no ties, so `n_nodes`,
   `density` and `mean_degree` silently exclude isolated people. In surveys, the isolated are
   often the point.

### Proposed behaviour

**1. Optional node table:** `--nodes nodes.csv` (`node_table=` in the API). It has a `node`
column, with the name configurable via `--node-col`, plus any attribute columns. An optional
`condition` column allows per-condition attributes. On load:
- Nodes listed in the table but absent from the edges are added as isolates, and the report
  says how many.
- Edge endpoints missing from the table are kept with empty attributes, and counted.
- Duplicate node ids are an error.

**2. Attribute mixing table** (`attribute_mixing.csv`). For each attribute named in the
profile's `node_attributes` (or `--attributes`), per condition:

| attribute | kind | n_groups | assortativity | same_group_tie_fraction | expected_same_group_fraction | p_value | community_nmi |
|---|---|---|---|---|---|---|---|

- `assortativity`: Newman's (2003) categorical or numeric assortativity coefficient.
- `same_group_tie_fraction`: the share of ties within a group. `expected_same_group_fraction`
  is what random mixing with the same group sizes and degrees would give. This is the plainest
  reading of homophily (McPherson et al., 2001).
- `p_value`: a permutation test that shuffles the attribute across nodes, with 1,000
  permutations and a seed. This matches the engine's habit of never quoting a statistic
  without its null.
- `community_nmi`: normalized mutual information between the detected communities and the
  attribute. It answers "do communities follow this attribute?" with one number, replacing
  the ad hoc misassignment count in the #58 test (which stays as the validation criterion).

Categorical vs numeric is inferred from the column type, and the profile can override it.
Directed networks use the directed versions (source attribute → target attribute).

**3. Profile.** `node_attributes` gets a role description per attribute, the same way
`signal_roles` works. Attribute-mixing interpretation text sits under `interpretation_rules`,
keyed `assortativity`, `same_group_tie_fraction`, and so on, and is validated like other
metric names.

### Validation

- **Karate club.** `community_nmi` against the factions, plus assortativity by faction with a
  permutation p. There's no published assortativity value to match, so this is a consistency
  check, not a known answer.
- **Political blogs** (Adamic & Glance, 2005): the known answer. 1,494 blogs, 759 liberal and
  735 conservative. "91% of the links originating within either the conservative or liberal
  communities stay within that community." The public `polblogs` network is a directed
  blogroll snapshot. The check: `same_group_tie_fraction` ≈ 0.91 for the directed network,
  with the tolerance and the handling of the dataset's small node-count difference fixed
  before running.
- **Isolates.** A unit test where a node-table-only node changes `n_nodes` and `density` by
  the expected amount.

---

## Delivery plan

Five PRs, each independently mergeable and each leaving existing outputs unchanged:

1. **Time axis and gap report.** Adds `date` as a time column, nominal-interval inference, and
   the gap/irregularity table. Reporting only; no metric changes.
2. **`gap_policy` (`join`/`segment`/`exclude`).** Segment-aware DFA, sample entropy and
   spectral entropy, plus the synthetic Ma et al. validation and the GHCN rerun with 23
   stations.
3. **`axis: time` and duration scales in `metric_parameters`,** plus declared value units in
   the report. The `climate_temperature` profile moves to `30D`/`3000D`.
4. **Node table ingestion.** Isolates and attribute validation, plus the karate factions moving
   from the side file into the pipeline.
5. **Attribute mixing metrics.** Permutation p-values and community NMI, plus the polblogs
   known-answer validation.

PRs 1–3 and 4–5 are independent tracks.

## Decisions (agreed 2026-10-10)

1. **Default `gap_policy`: `join`,** so no outputs change. It's sound for the positively
   correlated signals validated so far, and gaps are always reported. `segment` would
   be safer for unknown signals, but it changes the Fantasia and GHCN numbers.
2. **Irregular-sampling threshold: `spacing_cv > 0.1`,** as a default that a profile can override. This is a judgment call; there's no standard value.
3. **Permutation count: 1,000 by default.** That's fast for polblogs-sized networks (~1.5k
   nodes) but should be configurable for 100k-node networks.
4. **Value-unit conversion stays deferred** (#35 open), on the invariance argument above. Converting units now would mean designing without a real dataset that needs it.

## References

- Adamic, L. A. & Glance, N. (2005). The political blogosphere and the 2004 U.S. election: divided they blog. *Proc. 3rd Int. Workshop on Link Discovery*, 36–43.
- Ma, Q. D. Y., Bartsch, R. P., Bernaola-Galván, P., Yoneyama, M. & Ivanov, P. Ch. (2010). Effect of extreme data loss on long-range correlated and anticorrelated signals quantified by detrended fluctuation analysis. *Phys Rev E* 81:031101.
- McPherson, M., Smith-Lovin, L. & Cook, J. M. (2001). Birds of a feather: homophily in social networks. *Annu Rev Sociol* 27:415–444.
- Newman, M. E. J. (2003). Mixing patterns in networks. *Phys Rev E* 67:026126.
