# Validation on the Doubtful Sound dolphin network

Roadmap item A3 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This is the first real-data check for the `animal_social_network` profile.

## Data

`data/fetch_reference_networks.py` writes Lusseau's dolphin network (Newman's version, from
the Netzschleuder mirror) to `data/external/dolphins_edges.csv`, with each dolphin's name in
`dolphins_nodes.csv`. The run passes the names as a node table, so the per-node outputs carry them.
The network has 62 bottlenose dolphins, with a tie wherever a pair was seen together more often
than chance over 1994–2001.

## Criteria (fixed before the run)

The published values below were read from the papers' text:

- Lusseau & Newman (2004): 62 dolphins, 159 ties, all connected. Degree assortativity
  r = −0.044 ± 0.080, a null result unlike most human networks. SN100 has the highest
  betweenness, and its temporary disappearance coincided with the community splitting into
  two groups along the boundary found by community detection.
- Newman & Girvan (2004, Fig. 11): Q = 0.38 ± 0.08 for the two-group split, peaking at
  Q = 0.52 ± 0.03 when the larger group's four subgroups are included.

The test requires exact size, degree assortativity within 0.005 of −0.044, SN100 at the top of
betweenness, and modularity within 0.52 ± 0.03.

## Results

| check | published | engine | pass |
|---|---|---|---|
| nodes / ties / components | 62 / 159 / 1 | 62 / 159 / 1 | ✓ |
| degree assortativity | −0.044 | −0.0436 | ✓ |
| highest betweenness | SN100 | SN100 (0.248), then Beescratch (0.213) | ✓ |
| modularity | 0.52 ± 0.03 | 0.4955 | ✓, barely (limit 0.49) |

- The engine's modularity comes from greedy maximization (Clauset, Newman & Moore), not
  Newman & Girvan's betweenness method. It finds 4 communities (23, 22, 15 and 2 dolphins)
  against their 5, with Q 0.025 below theirs. It passes, but with only 0.0055 to spare, so the
  match is close rather than exact.
- The rest of the top five by betweenness (Beescratch, SN9, SN4, DN63) are all among the
  high-betweenness brokers highlighted in Lusseau & Newman's Fig. 2. That's an observation, not
  a criterion, since it comes from a figure.

## Not checked

- **Group membership and mixing by sex.** Both papers give these only in figures (vertex
  shapes and shades), and reading 62 labels off a figure would make an unreliable known answer.
  Lusseau & Newman's text does give the sex-mixing counts (58 male–male, 46 female–female and
  44 mixed ties, r = 0.346), so a sex table from a citable source would let the attribute-mixing
  work (design PR 5) reproduce r = 0.346 exactly.

## References

- Lusseau, D. & Newman, M. E. J. (2004). Identifying the role that animals play in their social networks. *Proc. R. Soc. Lond. B* 271:S477–S481 (arXiv q-bio/0403029).
- Newman, M. E. J. & Girvan, M. (2004). Finding and evaluating community structure in networks. *Phys. Rev. E* 69:026113.
