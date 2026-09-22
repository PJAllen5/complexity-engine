from pathlib import Path

import pandas as pd

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
