"""Node tables for the network pipeline (PR 4 of docs/design/time-axis-and-node-attributes.md):
isolates, attributes on per-node outputs, and validation of the table itself."""

import pandas as pd
import pytest

from complexity_engine.cli import network_main
from complexity_engine.network_analysis import analyze_network

TRIANGLE = [("a", "b"), ("b", "c"), ("c", "a")]


def _csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _edges(tmp_path, pairs=TRIANGLE, condition=None):
    rows = [{"source": s, "target": t, **({"condition": c} if condition else {})}
            for (s, t), c in zip(pairs, condition or [None] * len(pairs))]
    return _csv(tmp_path, "edges.csv", rows)


def _run(tmp_path, edges, nodes, **kwargs):
    return analyze_network(edges, node_table=nodes, weight_col=None, output_dir=tmp_path / "out", **kwargs)


def test_isolates_change_n_nodes_and_density_by_the_expected_amount(tmp_path):
    edges = _edges(tmp_path)
    nodes = _csv(tmp_path, "nodes.csv", [{"node": n, "group": g} for n, g in zip("abcde", "xxyyy")])
    without = analyze_network(edges, condition_col=None, weight_col=None, output_dir=tmp_path / "plain")
    result = _run(tmp_path, edges, nodes, condition_col=None)
    before, after = without["network_metrics_summary"].iloc[0], result["network_metrics_summary"].iloc[0]
    assert (before["n_nodes"], after["n_nodes"]) == (3, 5)
    # 3 edges among 3 nodes is complete (density 1); among 5 nodes it's 3 of 10 possible.
    assert (before["density"], after["density"]) == (1.0, pytest.approx(0.3))
    assert after["mean_degree"] == pytest.approx(6 / 5)
    assert after["n_connected_components"] == 3
    assert result["node_table_summary"].iloc[0]["isolates_added"] == 2
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Nodes" in report and "2 with no ties added as isolates" in report


def test_attributes_are_set_on_the_graph_and_joined_onto_per_node_outputs(tmp_path):
    nodes = _csv(tmp_path, "nodes.csv", [{"node": "a", "group": "x"}, {"node": "b", "group": "y"}])
    result = _run(tmp_path, _edges(tmp_path), nodes, condition_col=None,
                  compute_communities=True, compute_centrality=True)
    graph = result["graphs"]["all"]
    assert graph.nodes["a"]["group"] == "x" and "group" not in graph.nodes["c"]
    communities = result["node_communities"].set_index("node")
    assert communities.loc["b", "group"] == "y" and pd.isna(communities.loc["c", "group"])
    assert "group" in result["node_centrality"].columns
    assert result["node_table_summary"].iloc[0]["nodes_not_in_table"] == 1


def test_a_condition_column_gives_each_condition_its_own_nodes(tmp_path):
    edges = _edges(tmp_path, [("a", "b"), ("b", "c"), ("a", "b"), ("b", "c")], ["rest", "rest", "task", "task"])
    nodes = _csv(tmp_path, "nodes.csv", [
        {"condition": "rest", "node": "a", "role": "leader"},
        {"condition": "task", "node": "a", "role": "follower"},
        {"condition": "task", "node": "z", "role": "follower"},
        {"condition": "sleep", "node": "a", "role": "leader"},
    ])
    result = _run(tmp_path, edges, nodes)
    metrics = result["network_metrics_summary"].set_index("condition")
    assert (metrics.loc["rest", "n_nodes"], metrics.loc["task", "n_nodes"]) == (3, 4)
    assert result["graphs"]["rest"].nodes["a"]["role"] == "leader"
    assert result["graphs"]["task"].nodes["a"]["role"] == "follower"
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "1 node-table row(s) name a condition with no edges" in report


def test_a_table_without_conditions_applies_to_every_condition(tmp_path):
    edges = _edges(tmp_path, [("a", "b"), ("b", "c")], ["rest", "task"])
    nodes = _csv(tmp_path, "nodes.csv", [{"node": n} for n in "abc"])
    metrics = _run(tmp_path, edges, nodes)["network_metrics_summary"].set_index("condition")
    assert metrics["n_nodes"].to_dict() == {"rest": 3, "task": 3}


@pytest.mark.parametrize(
    "rows, kwargs, message",
    [
        ([{"node": "a"}, {"node": "a"}], {}, "lists some nodes more than once: a"),
        ([{"id": "a"}], {}, "has no 'node' column"),
    ],
)
def test_invalid_node_tables_are_rejected(tmp_path, rows, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, _edges(tmp_path), _csv(tmp_path, "nodes.csv", rows), condition_col=None, **kwargs)


def test_a_custom_node_id_column_is_accepted(tmp_path):
    nodes = _csv(tmp_path, "nodes.csv", [{"person": "d", "age": 40}])
    result = _run(tmp_path, _edges(tmp_path), nodes, condition_col=None, node_col="person")
    assert result["network_metrics_summary"].iloc[0]["n_nodes"] == 4
    assert result["graphs"]["all"].nodes["d"]["age"] == 40


def test_mismatched_id_types_are_an_error_not_a_graph_of_isolates(tmp_path):
    edges = _edges(tmp_path, [(1, 2), (2, 3)])
    nodes = tmp_path / "nodes.csv"
    nodes.write_text('node,group\n"n1",x\n', encoding="utf-8")  # unrelated ids: no error
    _run(tmp_path, edges, nodes, condition_col=None)
    nodes.write_text("node,group\n1.5,x\n", encoding="utf-8")  # unrelated numeric ids: no error either
    _run(tmp_path, edges, nodes, condition_col=None)
    from complexity_engine.network_analysis import _check_node_id_types
    with pytest.raises(ValueError, match="different types"):
        _check_node_id_types({"1", "2"}, {1, 2})


def test_cli_accepts_a_node_table(tmp_path, capsys):
    edges = _edges(tmp_path)
    nodes = _csv(tmp_path, "nodes.csv", [{"node": n} for n in "abcd"])
    network_main(["--data", str(edges), "--nodes", str(nodes), "--condition-col", "", "--weight-col", "",
                  "--output-dir", str(tmp_path / "cli")])
    assert "Added 1 node(s) from the node table with no ties, as isolates." in capsys.readouterr().out
