import pandas as pd

from complexity_engine.analysis import analyze_dataset


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
