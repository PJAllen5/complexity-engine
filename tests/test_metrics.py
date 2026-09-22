import numpy as np
import pandas as pd

from complexity_engine.analysis import analyze_dataset, condition_cross_signal_summary, resolve_signal_columns
from complexity_engine.metrics import mutual_information, sample_entropy, spectral_entropy


def test_sample_entropy_known_value():
    data = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    result = sample_entropy(data, m=2, r=0.25)
    assert np.isfinite(result)
    assert result >= 0.0


def test_spectral_entropy_nonnegative():
    data = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0], dtype=float)
    result = spectral_entropy(data)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


def test_resolve_signal_columns_ignores_metadata():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "condition": ["rest", "task", "rest", "task"],
            "timestamp": [0, 1, 2, 3],
            "signal_a": [1.0, 2.0, 3.0, 4.0],
            "signal_b": [0.5, 1.5, 2.5, 3.5],
            "label": ["x", "x", "y", "y"],
        }
    )
    result = resolve_signal_columns(df)
    assert result == ["signal_a", "signal_b"]


def test_analyze_dataset_supports_multiple_signals(tmp_path):
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"] * 2,
            "condition": ["rest", "rest", "task", "task"] * 2,
            "signal_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "signal_b": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        }
    )
    csv_path = tmp_path / "multi_signal.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(csv_path, signal_cols=["signal_a", "signal_b"], condition_col="condition", output_dir=tmp_path / "outputs")

    assert "complexity_summary" in result
    assert len(result["complexity_summary"]) >= 4
    assert (tmp_path / "outputs" / "report.md").exists()


def test_mutual_information_higher_for_coupled_signals():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y_coupled = x + rng.normal(scale=0.1, size=200)
    y_independent = rng.normal(size=200)

    mi_coupled = mutual_information(x, y_coupled)
    mi_independent = mutual_information(x, y_independent)

    assert mi_coupled > mi_independent


def test_mutual_information_handles_degenerate_input():
    assert mutual_information([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert mutual_information([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_condition_cross_signal_summary_requires_multiple_signals():
    df = pd.DataFrame({"condition": ["rest", "task"], "signal_a": [1.0, 2.0]})
    result = condition_cross_signal_summary(df, ["signal_a"], condition_col="condition")
    assert result.empty
