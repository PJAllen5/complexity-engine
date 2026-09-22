import pandas as pd

from complexity_engine.plots import (
    plot_complexity_metrics,
    plot_condition_comparison,
    plot_mutual_information_heatmap,
)


def _png_is_valid(path) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_condition_comparison_writes_valid_png(tmp_path):
    df = pd.DataFrame(
        {
            "condition": ["rest"] * 10 + ["task"] * 10,
            "signal": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.05, 0.95] + [5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 5.4, 4.6, 5.3, 4.7],
        }
    )
    output_path = tmp_path / "comparison.png"

    result = plot_condition_comparison(df, "signal", "condition", output_path)

    assert result == str(output_path)
    assert output_path.exists()
    assert _png_is_valid(output_path)


def test_plot_complexity_metrics_writes_valid_png(tmp_path):
    complexity_df = pd.DataFrame(
        {
            "signal": ["signal", "signal"],
            "condition": ["rest", "task"],
            "sample_entropy": [0.1, 0.9],
            "spectral_entropy": [0.8, 0.6],
            "dfa_exponent": [0.6, -0.1],
        }
    )
    output_path = tmp_path / "metrics.png"

    result = plot_complexity_metrics(
        complexity_df, "signal", ["sample_entropy", "spectral_entropy", "dfa_exponent"], output_path
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert _png_is_valid(output_path)


def test_plot_complexity_metrics_returns_none_when_no_data(tmp_path):
    empty_df = pd.DataFrame(columns=["signal", "condition", "sample_entropy"])
    result = plot_complexity_metrics(empty_df, "signal", ["sample_entropy"], tmp_path / "metrics.png")
    assert result is None


def test_plot_mutual_information_heatmap_writes_valid_png(tmp_path):
    cross_signal_df = pd.DataFrame(
        {
            "condition": ["rest", "task"],
            "signal_a": ["signal_a", "signal_a"],
            "signal_b": ["signal_b", "signal_b"],
            "mutual_information": [0.3, 0.5],
        }
    )
    output_path = tmp_path / "heatmap.png"

    result = plot_mutual_information_heatmap(cross_signal_df, output_path)

    assert result == str(output_path)
    assert output_path.exists()
    assert _png_is_valid(output_path)


def test_plot_mutual_information_heatmap_returns_none_when_empty(tmp_path):
    empty_df = pd.DataFrame(columns=["condition", "signal_a", "signal_b", "mutual_information"])
    result = plot_mutual_information_heatmap(empty_df, tmp_path / "heatmap.png")
    assert result is None
