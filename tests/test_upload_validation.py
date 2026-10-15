"""Tests for CSV upload validation and batch prediction."""

import pandas as pd
import pytest

from config.settings import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.predict import predict_batch_dataframe, validate_upload_schema
from src.train import train_classifier


@pytest.fixture
def trained_pipeline():
    """Fixture providing a mock trained pipeline."""
    X_train = pd.DataFrame(
        {
            "Type": ["M", "L", "H", "L"] * 5,
            "Air temperature [K]": [298.1, 299.0, 301.2, 298.5] * 5,
            "Process temperature [K]": [308.6, 309.1, 311.0, 308.9] * 5,
            "Rotational speed [rpm]": [1550, 1400, 1600, 1350] * 5,
            "Torque [Nm]": [42.8, 55.0, 30.0, 60.0] * 5,
            "Tool wear [min]": [0, 150, 20, 200] * 5,
        }
    )
    y_train = pd.Series([0, 1, 0, 1] * 5)
    return train_classifier(X_train, y_train, model_type="random_forest")


def test_upload_validation_success():
    """Verify schema validation passes when all required features exist."""
    valid_df = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [298.1],
            "Process temperature [K]": [308.6],
            "Rotational speed [rpm]": [1551],
            "Torque [Nm]": [42.8],
            "Tool wear [min]": [0],
        }
    )
    assert validate_upload_schema(valid_df) is True


def test_upload_validation_missing_columns():
    """Verify schema validation fails when required features are missing."""
    invalid_df = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [298.1],
        }
    )
    assert validate_upload_schema(invalid_df) is False


def test_upload_validation_invalid_input():
    """Verify schema validation returns False for None or non-DataFrames."""
    assert validate_upload_schema(None) is False
    assert validate_upload_schema("not_a_df") is False


def test_predict_batch_dataframe_success(trained_pipeline):
    """Verify batch prediction appends predicted_failure and failure_probability columns."""
    df_batch = pd.DataFrame(
        {
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 302.0],
            "Process temperature [K]": [308.6, 311.5],
            "Rotational speed [rpm]": [1551, 1350],
            "Torque [Nm]": [42.8, 65.0],
            "Tool wear [min]": [10, 220],
        }
    )

    result_df = predict_batch_dataframe(trained_pipeline, df_batch)
    assert len(result_df) == 2
    assert "predicted_failure" in result_df.columns
    assert "failure_probability" in result_df.columns
    assert "risk_level" in result_df.columns
    assert "risk_percentage" in result_df.columns
    assert result_df["predicted_failure"].isin([0, 1]).all()


def test_predict_batch_dataframe_missing_schema_raises_error(trained_pipeline):
    """Verify predict_batch_dataframe raises ValueError on invalid schema."""
    invalid_df = pd.DataFrame({"Type": ["M"]})
    with pytest.raises(ValueError, match="missing required features"):
        predict_batch_dataframe(trained_pipeline, invalid_df)
