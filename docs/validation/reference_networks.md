# Validation on Watts & Strogatz's small-world networks: C. elegans and the power grid

Roadmap items A1 and A2 ([validation-and-new-domains.md](../design/validation-and-new-domains.md)).
This is the first real-data check for the `neuronal_network` and `network` profiles. Both
networks are the ones Watts & Strogatz (1998) used to show that real networks are small worlds.

## Data

`data/fetch_reference_networks.py` downloads Mark Newman's versions from the Netzschleuder
mirror (networks.skewed.de), because the original umich.edu pages refuse scripted downloads,
and writes them to `data/external/` (not committed).

- **C. elegans neural network** (White et al., 1986): Newman's version has 297 neurons, not the
  paper's 282. It is directed and weighted by synapse count; it's written undirected and
  unweighted, without self-loops, as Watts & Strogatz analyzed it. That leaves 2,148 ties.
- **US western power grid**: 4,941 nodes and 6,594 lines, unweighted and undirected.

The same script also writes the dolphin network (A3) and political blogs (design PR 5), which
later validations use.

## Criteria (fixed before the run)

Published (Watts & Strogatz, 1998, Table 1):

| network | n | k | L | L_random | C | C_random |
|---|---|---|---|---|---|---|
| C. elegans | 282 | 14 | 2.65 | 2.25 | 0.28 | 0.05 |
| power grid | 4941 | 2.67 | 18.7 | 12.4 | 0.080 | 0.005 |

L is the average shortest path length and C the average clustering. The criteria compare with
the published values, and with a random graph's L ≈ ln n / ln k and C ≈ k / n, for each file's
own n and k. The exact thresholds are in `tests/test_reference_network_validation.py`.

## Results

| network | n | edges | k | C | L | C_random (k/n) | L_random (ln n/ln k) | sigma | omega |
|---|---|---|---|---|---|---|---|---|---|
| C. elegans | 297 | 2148 | 14.47 | 0.292 | 2.455 | 0.049 | 2.131 | 1.53 | 0.21 |
| power grid | 4941 | 6594 | 2.67 | 0.080 | 18.99 | 0.0005 | 8.66 | — | — |

- **Power grid:** the node and edge counts match exactly, and the network is one connected
  component. C = 0.0801 against the published 0.080, and L = 18.99 against 18.7 (+1.5%). The grid is
  about 150 times more clustered than a random graph. **Pass.**
- **Power grid, path length vs random: fail (strict xfail).** The criterion was L < 2 × L_random
  with L_random = ln n / ln k. That gives 8.66, far below the published 12.4. The formula
  assumes a tree-like random graph with many short paths. At k = 2.67, close to the percolation
  threshold, a random graph's giant component is stringy and its paths are much longer. The
  criterion was built on a wrong assumption about how Watts & Strogatz got their random values,
  and the power grid's C_random shows the same mismatch (k/n = 0.0005 vs the published 0.005).
  Against the published L_random of 12.4, the ratio is 1.53, which fits "close to random". That
  comparison was chosen after seeing the result, so it isn't counted as a pass.
- **C. elegans:** C = 0.292 against 0.28, and L = 2.455 against 2.65 (−7%, within the 10% set in
  advance for the 297-node version). It is 6 times more clustered than random, with L only 15%
  above random. sigma = 1.53 > 1. **Pass.**
- omega (Telesford et al., 2011) for C. elegans is +0.21, on the random-like side of 0. It
  wasn't a criterion. It's reported because sigma and omega disagree in strength: omega
  compares against a lattice as well, and the worm's clustering (0.29) is far below a
  lattice's.

## What this means for later validations

- For sparse networks (k below ~4), don't use ln n / ln k as a random baseline. Generate
  degree-matched random graphs instead, as sigma does, or compare with published random
  values. The large-network mode (E6) should keep this in mind for road networks, where k ≈ 2.8.
- sigma/omega on the 4,941-node grid weren't attempted, because they need repeated
  all-pairs paths on rewired copies. The C. elegans run, with both, takes most of this test's
  ~1 minute.

## References

- Humphries, M. D. & Gurney, K. (2008). Network 'small-world-ness': a quantitative method for determining canonical network equivalence. *PLoS ONE* 3:e0002051.
- Telesford, Q. K., Joyce, K. E., Hayasaka, S., Burdette, J. H. & Laurienti, P. J. (2011). The ubiquity of small-world networks. *Brain Connectivity* 1:367–375.
- Watts, D. J. & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature* 393:440–442.
- White, J. G., Southgate, E., Thomson, J. N. & Brenner, S. (1986). The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Phil. Trans. R. Soc. Lond. B* 314:1–340.
