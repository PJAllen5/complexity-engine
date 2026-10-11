# Validation on real social networks: karate club and ca-GrQc

This page records the first run of the network pipeline on real social data with published
results, using the `social_network` profile. It covers what was reproduced, what wasn't, and
where the edge-list data shape fell short of social network analysis.

## Data

`data/fetch_social_networks.py` writes both networks to `data/external/`, which is not
committed:

- **Zachary's karate club** (Zachary, 1977): 34 members, 78 friendship ties, observed before
  a dispute split the club in two. Each member's faction ("Mr. Hi" or "Officer", networkx's
  labels) goes to a separate node file, because an edge list has no place for it (see below).
  The ties are unweighted, as in the published analyses. The data ships with networkx.
- **ca-GrQc** (Leskovec, Kleinberg & Faloutsos, 2007): co-authorships among 5,242 authors in
  arXiv's General Relativity and Quantum Cosmology section, 1993–2003, from SNAP. Each tie is
  written once, and the 12 self-loops are kept because SNAP's edge count includes them.

## Method

```bash
python data/fetch_social_networks.py
python run_network_analysis.py --data data/external/karate_club_edges.csv --condition-col "" \
  --weight-col "" --domain social_network --compute-communities --output-dir outputs/karate
python run_network_analysis.py --data data/external/ca_grqc_edges.csv --condition-col "" \
  --weight-col "" --domain social_network --compute-communities --output-dir outputs/grqc
```

Communities come from the engine's existing method, greedy modularity maximization (Clauset,
Newman & Moore, 2004). The pass criteria were based on Newman & Girvan's (2004) results with
their own algorithms. Each detected community is mapped to its majority faction, and a member
counts as misassigned if their actual faction differs. The karate criteria were fixed before
the karate run: at most 1 member misassigned, and modularity between 0.35 and 0.45.

The ca-GrQc network was first run through the pipeline before its checks were written down.
Those checks compare against fixed published statistics, with no settings to choose, so
seeing the output first gave no room to tune them. The small-world thresholds (clustering
more than 100× the random estimate, path length under 1.5×) were set with the values in view.

## Results: karate club

| Measure | Engine | Published | Reproduced? |
|---|---|---|---|
| Modularity | 0.381 (3 communities) | ~0.4 at the two-community split (Newman & Girvan, 2004) | **Yes** |
| Members misassigned vs actual factions | 2 of 34 (members 9, 10) | 1 (betweenness method) or 0 (random-walk method) | **No**, by one member |

| Detected community | Mr. Hi | Officer |
|---|---|---|
| 0 (17 members) | 1 (member 9) | 16 |
| 1 (9 members) | 8 | 1 (member 10) |
| 2 (8 members) | 8 | 0 |

Greedy modularity splits Mr. Hi's faction in two. Its 0.381 matches the value published for
this algorithm on this network. Apart from two members, both communities nest inside the real
factions:

- **Member 9** is the one member Zachary's own flow model also got wrong. They had more ties to
  the officer's side but joined Mr. Hi's club, reportedly because they were weeks from a
  black-belt exam.
- **Member 10** has exactly two ties, one into each faction, so the structure alone can't
  place them.

Both errors are defensible, but the result misses the threshold fixed in advance. It's
recorded as a strict `xfail`, not as reproduced.

Not pre-specified, but consistent with the literature: the five members with the highest
betweenness are the two leaders (members 1 and 34) and three of their closest allies, which
fits the brokerage reading in the profile.

## Results: ca-GrQc

| Measure | Engine | Published | Reproduced? |
|---|---|---|---|
| Nodes, edges | 5,242, 14,496 | 5,242, 14,496 (SNAP) | **Yes**, exact |
| Average clustering | 0.5296 | 0.5296 (SNAP) | **Yes**, exact |
| Largest component | 4,158 nodes, 13,428 edges | 4,158, 13,428 (SNAP) | **Yes**, exact |
| Triangles | 48,260 | 48,260 (SNAP) | **Yes**, exact |
| Clustering vs random (k/n) | 0.53 vs 0.0016 (~340×) | far above random (Newman, 2001) | **Yes** |
| Path length vs random (ln n / ln k) | 6.05 vs 4.47 | comparable to random (Newman, 2001) | **Yes** |

The largest-component and triangle figures are computed in the test from the graph the
pipeline builds. The report doesn't show them.

Modularity is 0.82, with 417 communities across 355 connected components. No published value
was checked against it.

## What the run found in the engine

1. **Community assignments weren't output** (fixed). The pipeline computed a partition to
   score modularity, then discarded it. `--compute-communities` now writes each node's
   community to `node_communities.csv` and lists community sizes in the report. It's the
   same partition modularity scores, and a test checks that.

These were handled outside the engine for now. They are input for new data-shape design (#59):

2. **Edge lists can't carry node attributes.** Social network analysis is largely about how
   ties relate to who people are: homophily (McPherson et al., 2001), segregation, and
   whether communities follow known groups. The faction labels had to go in a separate file,
   joined in the test. Assortativity by attribute, or comparing communities with known groups,
   needs a node table alongside the edge list.
3. **Small-world statistics need a connected network.** `small_world_sigma`/`omega` are NaN
   for ca-GrQc, as for most real social networks, because it has 355 components. Supporting it
   would mean computing them on the giant component. They're also expensive at this size,
   since they rewire the network many times.
4. **Scale.** ca-GrQc takes about 40 s, almost all of it exact all-pairs path length. That's
   fine here, but networks 10–100× larger would need sampled path lengths.

## Reproducing

```bash
python data/fetch_social_networks.py
python -m pytest -q tests/test_social_network_validation.py
```

## References

- Clauset, A., Newman, M. E. J. & Moore, C. (2004). Finding community structure in very large networks. *Phys Rev E* 70:066111.
- Leskovec, J., Kleinberg, J. & Faloutsos, C. (2007). Graph evolution: densification and shrinking diameters. *ACM TKDD* 1(1):2. Data: [snap.stanford.edu/data/ca-GrQc](https://snap.stanford.edu/data/ca-GrQc.html)
- McPherson, M., Smith-Lovin, L. & Cook, J. M. (2001). Birds of a feather: homophily in social networks. *Annu Rev Sociol* 27:415–444.
- Newman, M. E. J. (2001). The structure of scientific collaboration networks. *PNAS* 98(2):404–409.
- Newman, M. E. J. & Girvan, M. (2004). Finding and evaluating community structure in networks. *Phys Rev E* 69:026113. [arXiv:cond-mat/0308217](https://arxiv.org/abs/cond-mat/0308217)
- Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. *J Anthropol Res* 33(4):452–473.
