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
