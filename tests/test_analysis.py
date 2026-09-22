from pathlib import Path

import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.metrics import compute_complexity_metrics


def _write_signal_csv(tmp_path, rest_values, task_values):
    df = pd.DataFrame(
        {
            "subject_id": ["s1"] * len(rest_values) + ["s2"] * len(task_values),
            "condition": ["rest"] * len(rest_values) + ["task"] * len(task_values),
            "signal": rest_values + task_values,
        }
    )
    csv_path = tmp_path / "signal.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_report_surfaces_analysis_question_and_signal_roles(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5], [10.0, 12.0, 9.0, 11.0])

    analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Analysis question:" in report
    assert "Domain signal roles" in report
    assert "primary_signal" in report


def test_report_warns_about_missing_expected_columns(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0], [3.0, 4.0])

    analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="cognitive",
        output_dir=tmp_path / "outputs",
    )

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "missing from this dataset: trial" in report


def test_interpretation_covers_all_recommended_metrics(tmp_path):
    rest_values = [1.0, 1.1, 1.0, 1.05] * 5
    task_values = [1.0, 5.0, 2.0, 8.0, 1.5, 6.0, 2.5, 9.0] * 3
    csv_path = _write_signal_csv(tmp_path, rest_values, task_values)

    analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "sample entropy is" in report
    assert "spectral entropy is" in report
    assert "dfa exponent is" in report


def test_only_domains_recommended_metrics_are_computed(tmp_path):
    config_path = tmp_path / "domains.yaml"
    config_path.write_text(
        """
minimal:
  name: minimal
  description: Minimal profile that only cares about sample entropy.
  recommended_metrics:
    - sample_entropy
  expected_columns:
    - signal
    - condition
""",
        encoding="utf-8",
    )
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5], [10.0, 12.0, 9.0, 11.0])

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="minimal",
        domain_config=config_path,
        output_dir=tmp_path / "outputs",
    )

    complexity = result["complexity_summary"]
    assert "sample_entropy" in complexity.columns
    assert "spectral_entropy" not in complexity.columns
    assert "dfa_exponent" not in complexity.columns


def test_compute_complexity_metrics_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown complexity metric"):
        compute_complexity_metrics([1.0, 2.0, 3.0], metrics=["not_a_real_metric"])


def test_compute_complexity_metrics_defaults_to_all_metrics():
    result = compute_complexity_metrics([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    assert set(result) == {"sample_entropy", "spectral_entropy", "dfa_exponent"}


def test_report_includes_entropy_complexity_caveat(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5], [10.0, 12.0, 9.0, 11.0])

    analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Methodological note" in report
    assert "Crutchfield & Young, 1989" in report


def test_cognitive_and_attention_interpretation_cites_predictive_processing(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5], [10.0, 12.0, 9.0, 11.0])

    for domain in ("cognitive", "attention"):
        analyze_dataset(
            csv_path,
            signal_cols=["signal"],
            condition_col="condition",
            domain=domain,
            output_dir=tmp_path / "outputs" / domain,
        )
        report = (tmp_path / "outputs" / domain / "report.md").read_text(encoding="utf-8")
        assert "Friston, 2010" in report


def _write_two_signal_csv(tmp_path):
    # signal_b uses a distinct (non-affine) pattern from signal_a so their entropy/DFA
    # metrics don't coincidentally match (these metrics are scale-invariant, so a signal
    # that's just a scaled/shifted copy of another would produce identical values).
    rest_a = [1.0, 1.1, 1.0, 1.05] * 5
    task_a = [1.0, 5.0, 2.0, 8.0, 1.5, 6.0, 2.5, 9.0] * 3
    rest_b = [2.0, 2.05, 2.15, 2.0, 2.1] * 4
    task_b = [4.0, 1.0, 3.5, 0.5, 4.5, 1.5, 3.0, 0.8] * 3

    df = pd.DataFrame(
        {
            "subject_id": ["s1"] * len(rest_a) + ["s2"] * len(task_a),
            "condition": ["rest"] * len(rest_a) + ["task"] * len(task_a),
            "signal_a": rest_a + task_a,
            "signal_b": rest_b + task_b,
        }
    )
    csv_path = tmp_path / "two_signal.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_multi_signal_report_groups_interpretation_by_signal(tmp_path):
    csv_path = _write_two_signal_csv(tmp_path)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal_a", "signal_b"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert "cross_signal_summary" in result
    assert not result["cross_signal_summary"].empty
    assert (tmp_path / "outputs" / "cross_signal_summary.csv").exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "### signal_a" in report
    assert "### signal_b" in report
    assert "## Cross-signal metrics" in report
    assert "### Cross-signal relationships" in report
    assert "mutual_information" in report


def test_single_signal_report_has_no_cross_signal_section(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5], [10.0, 12.0, 9.0, 11.0])

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert result["cross_signal_summary"].empty

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Cross-signal" not in report
    assert "### signal" not in report


def test_missing_columns_are_returned_programmatically(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0], [3.0, 4.0])

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="cognitive",
        output_dir=tmp_path / "outputs",
    )

    assert result["missing_columns"] == ["trial"]


def test_missing_columns_is_empty_when_all_expected_columns_present(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0], [3.0, 4.0])

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert result["missing_columns"] == []


def test_report_includes_statistical_significance(tmp_path):
    rest_values = [70.0, 71.0, 69.5, 70.5, 70.2] * 4
    task_values = [90.0, 91.0, 89.5, 90.5, 90.2] * 4
    csv_path = _write_signal_csv(tmp_path, rest_values, task_values)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    significance = result["significance_summary"]
    assert not significance.empty
    assert significance.iloc[0]["p_value"] < 0.001
    assert (tmp_path / "outputs" / "significance_summary.csv").exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Statistical significance" in report
    assert "differs significantly" in report
    assert "Welch's t-test" in report


def test_report_includes_visualizations(tmp_path):
    csv_path = _write_signal_csv(tmp_path, [1.0, 2.0, 1.5, 2.5] * 5, [10.0, 12.0, 9.0, 11.0] * 5)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert "signal_condition_comparison" in result["plot_paths"]
    assert "signal_complexity_metrics" in result["plot_paths"]
    for path in result["plot_paths"].values():
        assert (tmp_path / "outputs" / "plots").exists()
        assert Path(path).exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Visualizations" in report
    assert "![signal_condition_comparison](plots/signal_condition_comparison.png)" in report


def test_multi_signal_report_includes_mutual_information_heatmap(tmp_path):
    csv_path = _write_two_signal_csv(tmp_path)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal_a", "signal_b"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert "cross_signal_mutual_information" in result["plot_paths"]
    assert Path(result["plot_paths"]["cross_signal_mutual_information"]).exists()

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "![cross_signal_mutual_information](plots/cross_signal_mutual_information.png)" in report


def test_analyze_dataset_merges_multiple_files(tmp_path):
    df1 = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 10.0]})
    df2 = pd.DataFrame({"condition": ["rest", "task"], "signal": [2.0, 11.0]})
    path1 = tmp_path / "session1.csv"
    path2 = tmp_path / "session2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    result = analyze_dataset(
        [path1, path2],
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert result["provenance"]["input_rows"] == 4
    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "Data source: session1.csv, session2.csv" in report


def test_analyze_dataset_reports_provenance(tmp_path):
    df = pd.DataFrame(
        {
            "condition": ["rest", "rest", "task", None],
            "signal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    csv_path = tmp_path / "with_missing.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert result["provenance"]["input_rows"] == 4
    assert result["provenance"]["rows_dropped_missing_values"] == 1
    assert result["provenance"]["output_rows"] == 3

    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "## Data provenance" in report
    assert "Input rows: 4" in report
    assert "Rows analyzed: 3" in report
