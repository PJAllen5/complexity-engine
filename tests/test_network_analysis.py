from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.network_analysis import analyze_network


def _write_edge_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "edges.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_analyze_network_single_network_without_condition(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "a"},
        ],
    )

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    metrics = result["network_metrics_summary"]
    assert len(metrics) == 1
    assert metrics.iloc[0]["n_nodes"] == 3
    assert (tmp_path / "outputs" / "network_metrics_summary.csv").exists()
    assert (tmp_path / "outputs" / "report.md").exists()


def test_analyze_network_compares_conditions(tmp_path):
    rows = [{"source": "a", "target": "b", "condition": "rest"}, {"source": "b", "target": "c", "condition": "rest"}]
    rows += [
        {"source": "a", "target": "b", "condition": "task"},
        {"source": "b", "target": "c", "condition": "task"},
        {"source": "c", "target": "a", "condition": "task"},
    ]
    csv_path = _write_edge_csv(tmp_path, rows)

    result = analyze_network(csv_path, output_dir=tmp_path / "outputs")

    metrics = result["network_metrics_summary"]
    assert set(metrics["condition"]) == {"rest", "task"}

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "# Network Analysis Report" in report
    assert "## Network metrics" in report
    assert "in task than in rest" in report or "in rest than in task" in report


def test_analyze_network_report_includes_domain_grounding_and_visualization(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [
            {"source": "a", "target": "b", "condition": "rest"},
            {"source": "b", "target": "c", "condition": "rest"},
            {"source": "c", "target": "a", "condition": "rest"},
            {"source": "a", "target": "b", "condition": "task"},
            {"source": "a", "target": "c", "condition": "task"},
        ],
    )

    result = analyze_network(csv_path, output_dir=tmp_path / "outputs")

    assert result["degree_plot_path"] is not None
    assert Path(result["degree_plot_path"]).exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Watts & Strogatz, 1998" in report
    assert "Barabási & Albert, 1999" in report
    assert "![degree_distribution](plots/degree_distribution.png)" in report


def test_analyze_network_flags_missing_expected_columns(tmp_path):
    csv_path = _write_edge_csv(tmp_path, [{"source": "a", "target": "b"}])

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    assert result["missing_columns"] == ["condition"]


def test_analyze_network_directed_flag_changes_metrics_and_report(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "a"},
        ],
    )

    result = analyze_network(csv_path, condition_col=None, directed=True, output_dir=tmp_path / "outputs")

    metrics = result["network_metrics_summary"]
    assert "mean_in_degree" in metrics.columns
    assert "n_strongly_connected_components" in metrics.columns

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Directed: True" in report
    assert "strongly connected component" in report


def test_analyze_network_default_is_undirected_and_unaffected(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "c", "target": "a"}],
    )

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    metrics = result["network_metrics_summary"]
    assert "mean_in_degree" not in metrics.columns
    assert result["node_centrality"].empty
    assert result["centrality_csv"] is None
    assert result["node_communities"].empty
    assert result["communities_csv"] is None


def test_analyze_network_compute_centrality_writes_output(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "c", "target": "a"}],
    )

    result = analyze_network(csv_path, condition_col=None, compute_centrality=True, output_dir=tmp_path / "outputs")

    assert not result["node_centrality"].empty
    assert set(result["node_centrality"]["node"]) == {"a", "b", "c"}
    assert (tmp_path / "outputs" / "node_centrality.csv").exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Node centrality" in report


def test_analyze_network_rejects_domain_with_unknown_metric(tmp_path):
    csv_path = _write_edge_csv(tmp_path, [{"source": "a", "target": "b"}])

    config_path = tmp_path / "bad_domain.yaml"
    config_path.write_text(
        """
bad_network:
  name: bad_network
  description: Test domain with a metric that belongs to the time-series pipeline.
  recommended_metrics:
    - density
    - sample_entropy
  expected_columns: [source, target]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown recommended_metrics"):
        analyze_network(
            csv_path,
            condition_col=None,
            domain="bad_network",
            domain_config=config_path,
            output_dir=tmp_path / "outputs",
        )
    assert not (tmp_path / "outputs").exists()


def test_analyze_network_compute_small_world_adds_columns_and_report_section(tmp_path):
    import networkx as nx

    rest_graph = nx.watts_strogatz_graph(n=12, k=4, p=0.1, seed=1)
    task_graph = nx.barabasi_albert_graph(n=12, m=2, seed=1)
    rows = [{"source": f"n{u}", "target": f"n{v}", "condition": "rest"} for u, v in rest_graph.edges()]
    rows += [{"source": f"n{u}", "target": f"n{v}", "condition": "task"} for u, v in task_graph.edges()]
    csv_path = _write_edge_csv(tmp_path, rows)

    result = analyze_network(csv_path, compute_small_world=True, output_dir=tmp_path / "outputs")

    metrics = result["network_metrics_summary"]
    assert "small_world_sigma" in metrics.columns
    assert "small_world_omega" in metrics.columns
    assert metrics["small_world_sigma"].notna().all()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "### Small-world statistics" in report
    assert "Humphries & Gurney, 2008" in report
    assert "Telesford et al., 2011" in report


def test_analyze_network_small_world_disabled_by_default(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "c", "target": "a"}],
    )

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    assert "small_world_sigma" not in result["network_metrics_summary"].columns
    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Small-world statistics" not in report


def test_analyze_network_merges_multiple_files(tmp_path):
    df1 = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "condition": ["rest", "rest"]})
    df2 = pd.DataFrame({"source": ["c", "a"], "target": ["a", "d"], "condition": ["task", "task"]})
    path1 = tmp_path / "session1.csv"
    path2 = tmp_path / "session2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    result = analyze_network([path1, path2], output_dir=tmp_path / "outputs")

    assert result["provenance"]["input_rows"] == 4
    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Data source: session1.csv, session2.csv" in report


def test_analyze_network_reports_provenance(tmp_path):
    df = pd.DataFrame(
        {
            "source": ["a", None, "c"],
            "target": ["b", "c", "a"],
            "condition": ["rest", "rest", "rest"],
        }
    )
    csv_path = tmp_path / "with_missing.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    assert result["provenance"]["input_rows"] == 3
    assert result["provenance"]["rows_dropped_missing_values"] == 1
    assert result["provenance"]["output_rows"] == 2

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Data provenance" in report
    assert "Input rows: 3" in report
    assert "Rows analyzed: 2" in report


def test_analyze_network_detects_but_does_not_drop_duplicate_edges(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [
            {"source": "a", "target": "b", "condition": "rest"},
            {"source": "a", "target": "b", "condition": "rest"},
            {"source": "b", "target": "c", "condition": "rest"},
        ],
    )

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    assert result["provenance"]["duplicate_rows_detected"] == 1
    assert result["provenance"]["output_rows"] == 3  # not dropped


def test_analyze_network_writes_self_contained_html_report(tmp_path):
    csv_path = _write_edge_csv(
        tmp_path,
        [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "c", "target": "a"}],
    )

    result = analyze_network(csv_path, condition_col=None, output_dir=tmp_path / "outputs")

    html_path = Path(result["report_html"])
    assert html_path.exists()
    assert html_path == tmp_path / "outputs" / "report.html"
    html = html_path.read_text(encoding="utf-8")
    assert "<h1>Network Analysis Report</h1>" in html
    assert "data:image/png;base64," in html


def test_analyze_network_compute_communities_writes_output(tmp_path):
    rows = [("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x"), ("c", "x")]
    csv_path = _write_edge_csv(tmp_path, [{"source": s, "target": t} for s, t in rows])

    result = analyze_network(csv_path, condition_col=None, compute_communities=True, output_dir=tmp_path / "outputs")

    communities = result["node_communities"]
    assert set(communities["node"]) == set("abcxyz")
    assert communities["community"].nunique() == 2
    assert (tmp_path / "outputs" / "node_communities.csv").exists()
    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Communities" in report
    assert "- all: 2 communities, sizes 3, 3" in report
