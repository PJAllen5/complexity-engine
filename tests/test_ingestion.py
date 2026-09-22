import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import clean_dataset, load_dataset


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_dataset_single_file_has_unchanged_schema(tmp_path):
    df = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 2.0]})
    path = _write_csv(tmp_path, "single.csv", df)

    result = load_dataset(path)

    assert "source_file" not in result.columns
    assert list(result.columns) == ["condition", "signal"]


def test_load_dataset_merges_multiple_files_and_tags_source(tmp_path):
    df1 = pd.DataFrame({"condition": ["rest", "task"], "signal": [1.0, 2.0]})
    df2 = pd.DataFrame({"condition": ["rest", "task"], "signal": [3.0, 4.0]})
    path1 = _write_csv(tmp_path, "session1.csv", df1)
    path2 = _write_csv(tmp_path, "session2.csv", df2)

    result = load_dataset([path1, path2])

    assert result.shape[0] == 4
    assert set(result["source_file"]) == {"session1.csv", "session2.csv"}


def test_load_dataset_rejects_mismatched_columns(tmp_path):
    df1 = pd.DataFrame({"condition": ["rest"], "signal": [1.0]})
    df2 = pd.DataFrame({"condition": ["rest"], "other_col": [1.0]})
    path1 = _write_csv(tmp_path, "a.csv", df1)
    path2 = _write_csv(tmp_path, "b.csv", df2)

    with pytest.raises(ValueError, match="different columns"):
        load_dataset([path1, path2])


def test_load_dataset_rejects_empty_file_list():
    with pytest.raises(ValueError, match="No data files"):
        load_dataset([])


def test_clean_dataset_drops_rows_missing_required_values():
    df = pd.DataFrame(
        {
            "condition": ["rest", "rest", "task", None],
            "signal": [1.0, np.nan, 3.0, 4.0],
        }
    )

    cleaned, provenance = clean_dataset(df, required_cols=["signal", "condition"])

    assert cleaned.shape[0] == 2
    assert provenance["input_rows"] == 4
    assert provenance["rows_dropped_missing_values"] == 2
    assert provenance["output_rows"] == 2
    assert provenance["missing_value_columns_checked"] == ["signal", "condition"]


def test_clean_dataset_detects_duplicates_without_dropping_them():
    df = pd.DataFrame({"condition": ["rest", "rest", "rest"], "signal": [1.0, 1.0, 1.0]})

    cleaned, provenance = clean_dataset(df, required_cols=["signal", "condition"])

    assert provenance["duplicate_rows_detected"] == 2
    assert cleaned.shape[0] == 3  # not removed


def test_clean_dataset_ignores_required_columns_not_present():
    df = pd.DataFrame({"signal": [1.0, 2.0]})

    cleaned, provenance = clean_dataset(df, required_cols=["signal", "condition"])

    assert provenance["missing_value_columns_checked"] == ["signal"]
    assert cleaned.shape[0] == 2
