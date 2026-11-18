# Validation of attribute mixing: political blogs and the karate club

Design PR 5 ([time-axis-and-node-attributes.md](../design/time-axis-and-node-attributes.md)).
Attribute mixing measures homophily: whether ties form between nodes that share an attribute.

## Method

`attribute_mixing.csv` gets one row per attribute and condition. Only nodes with a value
count, and only ties between two such nodes; self-loops are excluded.

- **assortativity:** Newman's (2003) coefficient, categorical or numeric. It matches
  networkx's implementation exactly in the unit tests.
- **same_group_tie_fraction** and **expected_same_group_fraction:** the share of ties within a
  group, and the share random mixing would give with the same group sizes and degrees.
- **p_value:** a two-sided permutation test of assortativity (1,000 seeded shuffles by default).
- **community_nmi:** normalized mutual information between the attribute and the communities
  the modularity metric scores (Danon et al., 2005).

## Political blogs: the known answer

Adamic & Glance (2005) crawled the front pages of 1,494 political blogs (759 liberal, 735
conservative) in February 2005 and found that "91% of the links originating within either the
conservative or liberal communities stay within that community." The public `polblogs`
network is that link network: 1,490 blogs (758 liberal, 732 conservative) and 19,090 links,
of which 65 are repeats of an existing link and 3 are self-links. It's analyzed as directed, with repeats counted once.

**Criteria (fixed before the run):** directed same-side fraction within 0.91 ± 0.02;
assortativity positive with p < 0.05.

| measure | value |
|---|---|
| same-side tie fraction | **0.9115** (published 0.91) ✓ |
| expected under random mixing | 0.5004 |
| assortativity | 0.823, p = 0.001 (the minimum with 1,000 shuffles) ✓ |
| community NMI | 0.374 |

**Community NMI is held down by isolates.** 266 blogs have no links in this network.
Each is its own one-blog community in the greedy partition, which counts against agreement
with leaning. Over the 1,224 linked blogs, NMI is 0.634. The engine keeps the simpler
definition (all nodes with the attribute); this note is here so the value isn't misread.
NMI wasn't a criterion.

## Karate club: consistency check

There's no published assortativity for the karate factions, so this is a consistency check,
and its thresholds (assortativity > 0.5, NMI > 0.4) were set after seeing the values.
Assortativity by faction is 0.718 (p = 0.001), 85.9% of ties stay within a faction against
50.1% expected, and community NMI is 0.565. The factions now come in through the node table and
attribute mixing. The misassignment count fixed in #58 stays as the validation criterion,
as the design specified, and it's still a strict xfail.

## References

- Adamic, L. A. & Glance, N. (2005). The political blogosphere and the 2004 U.S. election: divided they blog. *Proc. 3rd Int. Workshop on Link Discovery*, 36–43.
- Danon, L., Díaz-Guilera, A., Duch, J. & Arenas, A. (2005). Comparing community structure identification. *J. Stat. Mech.* P09008.
- McPherson, M., Smith-Lovin, L. & Cook, J. M. (2001). Birds of a feather: homophily in social networks. *Annu. Rev. Sociol.* 27:415–444.
- Newman, M. E. J. (2003). Mixing patterns in networks. *Phys. Rev. E* 67:026126.
