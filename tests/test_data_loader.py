"""Focused offline tests for dataset loading, validation, and download error handling."""

import io
from pathlib import Path
import urllib.error
import zipfile
import pandas as pd
import pytest

from config.settings import (
    ALL_EXPECTED_COLUMNS,
    CATEGORICAL_FEATURES,
    FAILURE_TYPE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from scripts.download_data import download_dataset
from src.data_loader import load_raw_data, validate_raw_data


def create_valid_dummy_df(num_rows: int = 5) -> pd.DataFrame:
    """Helper to create a minimal, structurally valid dummy AI4I DataFrame."""
    return pd.DataFrame(
        {
            "UDI": list(range(1, num_rows + 1)),
            "Product ID": [f"M1486{i}" for i in range(num_rows)],
            "Type": ["M"] * num_rows,
            "Air temperature [K]": [298.1 + i * 0.1 for i in range(num_rows)],
            "Process temperature [K]": [308.6 + i * 0.1 for i in range(num_rows)],
            "Rotational speed [rpm]": [1550 + i * 10 for i in range(num_rows)],
            "Torque [Nm]": [42.8 - i * 0.5 for i in range(num_rows)],
            "Tool wear [min]": [i * 5 for i in range(num_rows)],
            "Machine failure": [0] * (num_rows - 1) + [1],
            "TWF": [0] * num_rows,
            "HDF": [0] * num_rows,
            "PWF": [0] * (num_rows - 1) + [1],
            "OSF": [0] * num_rows,
            "RNF": [0] * num_rows,
        }
    )


def test_load_raw_data_success(tmp_path: Path):
    """Test loading a valid CSV file."""
    valid_csv = tmp_path / "test_data.csv"
    df_expected = create_valid_dummy_df(5)
    df_expected.to_csv(valid_csv, index=False)

    df_loaded = load_raw_data(valid_csv, validate=True)
    assert len(df_loaded) == 5
    assert list(df_loaded.columns) == ALL_EXPECTED_COLUMNS
    assert df_loaded[TARGET_COLUMN].tolist() == [0, 0, 0, 0, 1]


def test_load_raw_data_with_bom(tmp_path: Path):
    """Test loading a CSV file containing a UTF-8 Byte Order Mark (BOM)."""
    bom_csv = tmp_path / "bom_data.csv"
    df_expected = create_valid_dummy_df(3)
    # Write using utf-8-sig to include BOM
    df_expected.to_csv(bom_csv, index=False, encoding="utf-8-sig")

    df_loaded = load_raw_data(bom_csv, validate=True)
    # Ensure UDI is not corrupted into \ufeffUDI
    assert "UDI" in df_loaded.columns
    assert len(df_loaded) == 3


def test_load_raw_data_missing_file(tmp_path: Path):
    """Test that missing file raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(missing_file)


def test_load_raw_data_empty_file(tmp_path: Path):
    """Test that an empty 0-byte file raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
        load_raw_data(empty_file)


def test_validate_raw_data_empty_dataframe():
    """Test that empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_raw_data(pd.DataFrame())


def test_validate_raw_data_missing_columns():
    """Test that missing expected columns raises ValueError."""
    df = create_valid_dummy_df(3)
    df_missing = df.drop(columns=["Machine failure", "Rotational speed [rpm]"])

    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        validate_raw_data(df_missing)


def test_validate_raw_data_invalid_target_values():
    """Test that non-binary target values raise ValueError."""
    df = create_valid_dummy_df(4)
    df.loc[0, TARGET_COLUMN] = 99  # Invalid class label

    with pytest.raises(ValueError, match="must only contain binary values"):
        validate_raw_data(df)


def test_validate_raw_data_target_nulls():
    """Test that null values in target raise ValueError."""
    df = create_valid_dummy_df(4)
    df.loc[0, TARGET_COLUMN] = None

    with pytest.raises(ValueError, match="contains null or missing values"):
        validate_raw_data(df)


def test_validate_raw_data_invalid_type_category():
    """Test that invalid categories in 'Type' column raise ValueError."""
    df = create_valid_dummy_df(3)
    df.loc[0, "Type"] = "Z"  # Invalid type (only L, M, H are allowed)

    with pytest.raises(ValueError, match="contains invalid categories"):
        validate_raw_data(df)


def test_validate_raw_data_non_numeric_features():
    """Test that non-numeric numerical features raise ValueError."""
    df = create_valid_dummy_df(3)
    df["Torque [Nm]"] = ["high", "medium", "low"]

    with pytest.raises(ValueError, match="must have a numeric dtype"):
        validate_raw_data(df)


def test_validate_raw_data_invalid_failure_modes():
    """Test that non-binary failure-mode columns raise ValueError."""
    df = create_valid_dummy_df(3)
    df.loc[0, "TWF"] = 3

    with pytest.raises(ValueError, match="Failure-mode column 'TWF' must be binary"):
        validate_raw_data(df)


def test_download_dataset_offline_mock(monkeypatch, tmp_path: Path):
    """Test download_dataset extracting and validating via mocked network response."""
    # Build in-memory zip archive containing valid dummy ai4i2020.csv
    dummy_df = create_valid_dummy_df(10)
    csv_bytes = dummy_df.to_csv(index=False).encode("utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("ai4i2020.csv", csv_bytes)
    zip_bytes = zip_buffer.getvalue()

    class MockResponse:
        status = 200
        reason = "OK"

        def read(self):
            return zip_bytes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=30: MockResponse()
    )

    dest_file = tmp_path / "ai4i2020.csv"
    result_path = download_dataset(
        url="http://mock-uci.local/data.zip",
        target_file=dest_file,
        force=True,
    )

    assert result_path == dest_file
    assert dest_file.exists()
    loaded_df = load_raw_data(dest_file, validate=True)
    assert len(loaded_df) == 10


def test_download_dataset_network_failure_mock(monkeypatch, tmp_path: Path):
    """Test download failure raises RuntimeError and does not generate synthetic files."""
    def mock_urlopen_fail(req, timeout=30):
        raise urllib.error.URLError("Simulated connection timeout")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)

    dest_file = tmp_path / "ai4i2020.csv"
    with pytest.raises(RuntimeError, match="Failed to download dataset"):
        download_dataset(
            url="http://mock-uci.local/data.zip",
            target_file=dest_file,
            force=True,
        )

    # Ensure no synthetic replacement file is created
    assert not dest_file.exists()
