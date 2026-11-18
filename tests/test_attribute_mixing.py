"""Attribute mixing on networks (design PR 5 in docs/design/time-axis-and-node-attributes.md)."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from complexity_engine.cli import network_main
from complexity_engine.domain_profiles import DomainProfile
from complexity_engine.network_analysis import analyze_network
from complexity_engine.network_metrics import attribute_mixing, normalized_mutual_information


def _karate(directed=False):
    g = nx.karate_club_graph()
    if directed:
        d = nx.DiGraph(list(g.edges()))
        nx.set_node_attributes(d, nx.get_node_attributes(g, "club"), "club")
        return d
    return g


@pytest.mark.parametrize("directed", [False, True])
def test_categorical_assortativity_matches_networkx(directed):
    g = _karate(directed)
    assert attribute_mixing(g, "club", permutations=19)["assortativity"] == pytest.approx(
        nx.attribute_assortativity_coefficient(g, "club")
    )


def test_numeric_assortativity_matches_networkx():
    g = _karate()
    nx.set_node_attributes(g, {n: float(n % 7) for n in g}, "x")
    assert attribute_mixing(g, "x", kind="numeric", permutations=19)["assortativity"] == pytest.approx(
        nx.numeric_assortativity_coefficient(g, "x")
    )


def test_two_separate_groups_are_perfectly_assortative():
    # Two groups of 6: only 2 of the C(12, 6) = 924 shufflings separate them perfectly, so p ~ 0.002.
    g = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    nx.set_node_attributes(g, {n: "a" if n < 6 else "b" for n in g}, "group")
    result = attribute_mixing(g, "group", permutations=99)
    assert result["assortativity"] == pytest.approx(1.0)
    assert result["same_group_tie_fraction"] == 1.0 and result["expected_same_group_fraction"] == 0.5
    assert result["community_nmi"] == pytest.approx(1.0)
    assert result["p_value"] < 0.05


def test_a_bipartite_graph_is_disassortative():
    g = nx.complete_bipartite_graph(4, 4)
    result = attribute_mixing(g, "bipartite", permutations=99)
    assert result["assortativity"] == pytest.approx(-1.0) and result["same_group_tie_fraction"] == 0.0


def test_unlabelled_nodes_and_self_loops_are_left_out():
    g = nx.Graph([("a", "b"), ("b", "c"), ("c", "c"), ("c", "d")])
    nx.set_node_attributes(g, {"a": "x", "b": "x", "c": "y"}, "group")
    result = attribute_mixing(g, "group", permutations=19)
    assert result["n_nodes"] == 3
    assert result["same_group_tie_fraction"] == 0.5  # a-b same, b-c different; c-c and c-d excluded


def test_permutation_p_is_seeded():
    g = _karate()
    runs = [attribute_mixing(g, "club", permutations=50, seed=s)["p_value"] for s in (1, 1)]
    assert runs[0] == runs[1]
    g2 = _karate()
    nx.set_node_attributes(g2, {n: str(np.random.default_rng(n).integers(2)) for n in g2}, "noise")
    assert attribute_mixing(g2, "noise", permutations=199, seed=3)["p_value"] > 0.05


def test_nmi_bounds():
    assert normalized_mutual_information([0, 0, 1, 1], ["x", "x", "y", "y"]) == pytest.approx(1.0)
    assert normalized_mutual_information([0, 1, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "spec, message",
    [("not a mapping", "node_attributes must map"), ({"x": {"kind": "ordinal"}}, "kind must be one of"),
     ({"x": {"role": "r", "scale": 2}}, "a mapping with role and kind")],
)
def test_profile_rejects_invalid_node_attributes(spec, message):
    with pytest.raises(ValueError, match=message):
        DomainProfile.from_mapping({"name": "t", "node_attributes": spec})


def test_profile_accepts_a_role_string_or_a_mapping():
    profile = DomainProfile.from_mapping(
        {"name": "t", "node_attributes": {"sex": "Recorded sex.", "age": {"role": "Years.", "kind": "numeric"}}}
    )
    assert profile.node_attributes == {"sex": {"role": "Recorded sex.", "kind": ""},
                                       "age": {"role": "Years.", "kind": "numeric"}}


def _files(tmp_path):
    edges = tmp_path / "edges.csv"
    pd.DataFrame({"source": list("aabcdde"), "target": list("bcceeff")}).to_csv(edges, index=False)
    nodes = tmp_path / "nodes.csv"
    pd.DataFrame({"node": list("abcdef"), "team": list("xxxyyy"), "score": [1, 2, 3, 7, 8, 9],
                  "flag": [0, 0, 0, 1, 1, 1]}).to_csv(nodes, index=False)
    return edges, nodes


def test_pipeline_writes_attribute_mixing_and_infers_kinds(tmp_path):
    edges, nodes = _files(tmp_path)
    result = analyze_network(edges, node_table=nodes, attributes=["team", "score", "flag"], condition_col=None,
                             weight_col=None, permutations=99, output_dir=tmp_path / "out")
    mixing = result["attribute_mixing"].set_index("attribute")
    assert mixing["kind"].to_dict() == {"team": "categorical", "score": "numeric", "flag": "categorical"}
    assert mixing.loc["team", "assortativity"] == pytest.approx(mixing.loc["flag", "assortativity"])
    assert (tmp_path / "out" / "attribute_mixing.csv").exists()
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Attribute mixing" in report and "permutation test of assortativity (99 shuffles" in report


def test_attributes_need_a_node_table_and_existing_columns(tmp_path):
    edges, nodes = _files(tmp_path)
    with pytest.raises(ValueError, match="needs a node table"):
        analyze_network(edges, attributes=["team"], condition_col=None, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="no column\\(s\\) height"):
        analyze_network(edges, node_table=nodes, attributes=["height"], condition_col=None, output_dir=tmp_path / "out")


def test_profile_node_attributes_drive_mixing_and_can_set_kind(tmp_path):
    edges, nodes = _files(tmp_path)
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "mixing_test:\n  name: mixing_test\n  description: Test.\n  recommended_metrics: [n_nodes]\n"
        "  node_attributes:\n    score: {role: A score., kind: categorical}\n", encoding="utf-8")
    result = analyze_network(edges, node_table=nodes, condition_col=None, weight_col=None, domain="mixing_test",
                             domain_config=profile, permutations=19, output_dir=tmp_path / "out")
    row = result["attribute_mixing"].iloc[0]
    assert (row["attribute"], row["kind"], row["n_groups"]) == ("score", "categorical", 6)
    assert "- score: A score." in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_cli_accepts_attributes(tmp_path, capsys):
    edges, nodes = _files(tmp_path)
    network_main(["--data", str(edges), "--nodes", str(nodes), "--attributes", "team", "--permutations", "19",
                  "--condition-col", "", "--weight-col", "", "--output-dir", str(tmp_path / "cli")])
    assert "Saved attribute mixing to:" in capsys.readouterr().out
