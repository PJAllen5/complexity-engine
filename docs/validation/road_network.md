# Scale test: California's road network (roadNet-CA)

Roadmap items E6 (large-network mode) and B5 (`infrastructure_network` profile),
[validation-and-new-domains.md](../design/validation-and-new-domains.md).

## What large-network mode changes

Before this, every network got exact all-pairs shortest paths and greedy modularity, and the
graph was built one edge at a time. On roadNet-CA's 2 million nodes none of that finishes: one
breadth-first search takes about 2.7 s, so exact paths would take weeks.

- The graph is built in bulk: 4 s for 2.7 million edges.
- Above 50,000 nodes in the largest component, the average shortest path is estimated from
  `--path-sources` (default 50) randomly chosen sources. Each source's distances are exact.
- Above 100,000 nodes, communities come from Louvain (Blondel et al., 2008), seeded, in about
  2 minutes here.
- Small-world statistics are skipped above 5,000 nodes.

Networks below those sizes are analyzed exactly as before. All validated networks so far are
below them, and their results are unchanged.

## Data

`data/fetch_road_network.py` downloads SNAP's roadNet-CA (Leskovec et al., 2009). Its
intersections and road endpoints are nodes and road segments are edges, written once per pair
(about 40 MB, not committed).

## Criteria and results

The SNAP statistics were the criteria, fixed before the pipeline ran. networkx's own clustering
and triangle count were seen while checking feasibility, so those two weren't blind.

| criterion | published | engine |
|---|---|---|
| nodes / edges | 1,965,206 / 2,766,607 | 1,965,206 / 2,766,607 ✓ |
| largest component | 1,957,027 | 1,957,027 ✓ |
| average clustering | 0.0464 | 0.04637 ✓ |
| triangles | 120,676 | 120,676 ✓ |
| mean path below the 90-percentile effective diameter | < 500 | 317.9 (20 sources) ✓ |
| full run within 10 minutes | | 210 s ✓ |
| 20-source estimate stable across seeds (±5%) | | **317.9 vs 297.5, 6.4% ✗** (strict xfail) |

Also reported, though not criteria: mean degree 2.82, 2,638 connected components, transitivity
0.060 (SNAP's "fraction of closed triangles", 0.021, is computed differently), and Louvain
modularity 0.992. Geography keeps nearly all road links local.

## What failed

Twenty sources weren't enough for ±5%. A third 20-source sample gave 320.2, so the three
estimates span 297–320. In a spatial network a source's mean distance depends strongly on where
it sits: a junction at the edge of the state is far from everything. Sampling error shrinks with
the square root of the number of sources, so the default of 50 should roughly halve the spread
(about ±3%). That wasn't tested here, since each source costs about 2.7 s.

## References

- Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.* P10008.
- Leskovec, J., Lang, K. J., Dasgupta, A. & Mahoney, M. W. (2009). Community structure in large networks: natural cluster sizes and the absence of large well-defined clusters. *Internet Mathematics* 6:29–123.
