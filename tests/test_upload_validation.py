"""Tests for CSV upload validation and batch prediction."""

import pandas as pd
import pytest

from config.settings import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from src.predict import (
    evaluate_batch_predictions,
    extract_actual_failure_column,
    predict_batch_dataframe,
    validate_upload_schema,
)
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
    return train_classifier(X_train, y_train, model_type="decision_tree")


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


def test_predict_batch_dataframe_preserves_actual_labels(trained_pipeline):
    """Verify batch predictions preserve existing ground truth and leakage columns without collision."""
    df_with_labels = pd.DataFrame(
        {
            "UDI": [101, 102],
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 302.0],
            "Process temperature [K]": [308.6, 311.5],
            "Rotational speed [rpm]": [1551, 1350],
            "Torque [Nm]": [42.8, 65.0],
            "Tool wear [min]": [10, 220],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
        }
    )

    result_df = predict_batch_dataframe(trained_pipeline, df_with_labels)
    assert "Machine failure" in result_df.columns
    assert "predicted_failure" in result_df.columns
    assert list(result_df["Machine failure"]) == [0, 1]
    assert "TWF" in result_df.columns


def test_extract_actual_failure_column():
    """Verify detection of ground-truth failure columns."""
    df_with_target = pd.DataFrame({"Machine failure": [0, 1], "Type": ["M", "L"]})
    assert extract_actual_failure_column(df_with_target) == "Machine failure"

    df_with_alias = pd.DataFrame({"Actual_Failure": [1, 0], "Type": ["M", "L"]})
    assert extract_actual_failure_column(df_with_alias) == "Actual_Failure"

    df_without_target = pd.DataFrame({"Type": ["M", "L"], "Torque [Nm]": [40.0, 50.0]})
    assert extract_actual_failure_column(df_without_target) is None

    assert extract_actual_failure_column(None) is None


def test_evaluate_batch_predictions():
    """Verify separate actual-versus-predicted metric calculations."""
    annotated = pd.DataFrame(
        {
            "Machine failure": [1, 1, 0, 0],
            "predicted_failure": [1, 0, 1, 0],
            "failure_probability": [0.9, 0.2, 0.75, 0.05],
        }
    )
    metrics = evaluate_batch_predictions(annotated, "Machine failure")

    assert metrics["total_samples"] == 4
    assert metrics["actual_failures"] == 2
    assert metrics["predicted_failures"] == 2
    assert metrics["tp"] == 1  # 1st sample
    assert metrics["fn"] == 1  # 2nd sample
    assert metrics["fp"] == 1  # 3rd sample
    assert metrics["tn"] == 1  # 4th sample
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5

    with pytest.raises(ValueError, match="not found"):
        evaluate_batch_predictions(annotated, "Nonexistent_Col")
