import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.domain_profiles import DomainProfile, validate_metric_parameters, validate_recommended_metrics
from complexity_engine.metrics import dfa_exponent


def test_domain_profile_is_applied(tmp_path):
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"] * 2,
            "condition": ["rest", "rest", "task", "task"] * 2,
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    csv_path = tmp_path / "domain.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(
        csv_path,
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )

    assert result["domain_profile"]["name"] == "physiology"
    report = (tmp_path / "outputs" / "report.md").read_text(encoding="utf-8")
    assert "physiology" in report.lower()
    assert "sample entropy" in report.lower()


def test_unknown_domain_rejected(tmp_path):
    df = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 2.0]})
    csv_path = tmp_path / "bad_domain.csv"
    df.to_csv(csv_path, index=False)

    try:
        analyze_dataset(csv_path, signal_cols=["signal"], domain="unknown", output_dir=tmp_path / "outputs")
        assert False, "Expected ValueError for unknown domain"
    except ValueError:
        pass


def test_extended_domain_metadata_is_available(tmp_path):
    profile = analyze_dataset(
        __import__("pathlib").Path("data/physio_example.csv"),
        signal_cols=["signal"],
        condition_col="condition",
        domain="physiology",
        output_dir=tmp_path / "outputs",
    )["domain_profile"]

    assert "expected_columns" in profile
    assert "analysis_question" in profile
    assert "primary_signal" in profile["signal_roles"]


def test_domain_profiles_can_be_loaded_from_json(tmp_path):
    config_path = tmp_path / "domains.json"
    config_path.write_text(
        '{"stress": {"name": "stress", "description": "Stress monitoring profile", "analysis_question": "Does task stress elevate irregularity?", "recommended_metrics": ["sample_entropy", "spectral_entropy"], "expected_columns": ["signal", "condition"], "signal_roles": {"primary_signal": "Physiological marker", "condition": "Task condition"}, "interpretation_rules": {"sample_entropy": "Higher values may indicate greater variability."}}}',
        encoding="utf-8",
    )

    from complexity_engine.domain_profiles import load_domain_profiles

    profiles = load_domain_profiles(config_path)
    assert "stress" in profiles
    assert profiles["stress"].analysis_question == "Does task stress elevate irregularity?"


def test_domain_profiles_can_be_loaded_from_yaml(tmp_path):
    config_path = tmp_path / "domains.yaml"
    config_path.write_text(
        """
stress:
  name: stress
  description: Stress monitoring profile
  analysis_question: Does task stress elevate irregularity?
  recommended_metrics:
    - sample_entropy
    - spectral_entropy
  expected_columns:
    - signal
    - condition
  signal_roles:
    primary_signal: Physiological marker
    condition: Task condition
  interpretation_rules:
    sample_entropy: Higher values may indicate greater variability.
""",
        encoding="utf-8",
    )

    from complexity_engine.domain_profiles import load_domain_profiles

    profiles = load_domain_profiles(config_path)
    assert "stress" in profiles
    assert profiles["stress"].expected_columns == ("signal", "condition")


def test_validate_recommended_metrics_accepts_known_metrics():
    profile = DomainProfile(name="test", description="", recommended_metrics=("sample_entropy",))
    validate_recommended_metrics(profile, {"sample_entropy", "spectral_entropy"})  # should not raise


def test_validate_recommended_metrics_rejects_unknown_metric():
    profile = DomainProfile(name="test", description="", recommended_metrics=("sample_entropy", "modularity"))
    with pytest.raises(ValueError, match="unknown recommended_metrics"):
        validate_recommended_metrics(profile, {"sample_entropy", "spectral_entropy"})


def test_analyze_dataset_rejects_domain_with_unknown_metric(tmp_path):
    df = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 2.0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    config_path = tmp_path / "bad_domain.yaml"
    config_path.write_text(
        """
bad_domain:
  name: bad_domain
  description: Test domain with a metric that belongs to the network pipeline.
  recommended_metrics:
    - sample_entropy
    - modularity
  expected_columns: [signal, condition]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown recommended_metrics"):
        analyze_dataset(
            csv_path,
            signal_cols=["signal"],
            domain="bad_domain",
            domain_config=config_path,
            output_dir=tmp_path / "outputs",
        )
    # Fails before any output is written -- this is a load-time check, not a
    # mid-analysis crash.
    assert not (tmp_path / "outputs").exists()


_ACCEPTED = {"dfa_exponent": ("min_scale", "max_scale", "order"), "sample_entropy": ("m", "r")}


def _profile_with_parameters(metric_parameters):
    return DomainProfile.from_mapping(
        {"name": "test", "recommended_metrics": ["dfa_exponent"], "metric_parameters": metric_parameters}
    )


def test_validate_metric_parameters_accepts_known_parameters():
    profile = _profile_with_parameters({"dfa_exponent": {"min_scale": 30, "order": 2}})
    validate_metric_parameters(profile, _ACCEPTED)  # should not raise
    assert profile.as_dict()["metric_parameters"] == {"dfa_exponent": {"min_scale": 30, "order": 2}}


@pytest.mark.parametrize(
    "metric_parameters, message",
    [
        ({"dfa_exponnt": {"order": 2}}, "unknown metric 'dfa_exponnt'"),
        ({"sample_entropy": {"m": 3}}, "isn't in its recommended_metrics"),
        ({"dfa_exponent": {"max_scal": 3000}}, r"unknown parameter\(s\) \['max_scal'\]"),
    ],
)
def test_validate_metric_parameters_rejects_mistakes(metric_parameters, message):
    with pytest.raises(ValueError, match=message):
        validate_metric_parameters(_profile_with_parameters(metric_parameters), _ACCEPTED)


def test_metric_parameters_must_be_nested_mappings():
    with pytest.raises(ValueError, match="metric_parameters must map"):
        _profile_with_parameters({"dfa_exponent": 30})


def test_analyze_dataset_applies_and_reports_metric_parameters(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "condition": np.repeat(["a", "b"], 2000),
            "signal": rng.standard_normal(4000),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    config_path = tmp_path / "domain.yaml"
    config_path.write_text(
        """
scaled:
  name: scaled
  description: Test domain with a configured DFA range.
  recommended_metrics: [dfa_exponent]
  metric_parameters:
    dfa_exponent: {min_scale: 10, max_scale: 100, order: 2}
""",
        encoding="utf-8",
    )

    result = analyze_dataset(
        csv_path, signal_cols=["signal"], domain="scaled", domain_config=config_path, output_dir=tmp_path / "out"
    )

    expected = dfa_exponent(df.loc[df["condition"] == "a", "signal"].to_numpy(), min_scale=10, max_scale=100, order=2)
    complexity = result["complexity_summary"].set_index("condition")
    assert complexity.loc["a", "dfa_exponent"] == pytest.approx(expected)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "- Metric settings: dfa_exponent(min_scale=10, max_scale=100, order=2)" in report
