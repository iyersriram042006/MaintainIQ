"""Tests for inference and failure risk prediction pipeline."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.predict import assess_risk_level, predict_failure_risk
from src.train import train_classifier


@pytest.fixture
def trained_pipeline() -> Pipeline:
    """Fixture providing a small trained Decision Tree pipeline."""
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


def test_assess_risk_level():
    """Verify operational risk level thresholds."""
    assert assess_risk_level(0.85) == "High"
    assert assess_risk_level(0.70) == "High"
    assert assess_risk_level(0.50) == "Medium"
    assert assess_risk_level(0.30) == "Medium"
    assert assess_risk_level(0.15) == "Low"
    assert assess_risk_level(0.0) == "Low"


def test_predict_failure_risk_single_dict(trained_pipeline):
    """Verify prediction output structure for a single feature dictionary."""
    sample = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 5,
    }

    result = predict_failure_risk(trained_pipeline, sample)
    assert isinstance(result, dict)
    assert "machine_failure_prediction" in result
    assert result["machine_failure_prediction"] in [0, 1]
    assert "failure_probability" in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert "risk_percentage" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_predict_failure_risk_dataframe_batch(trained_pipeline):
    """Verify batch predictions on a DataFrame."""
    df_sample = pd.DataFrame(
        {
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 299.5],
            "Process temperature [K]": [308.6, 309.8],
            "Rotational speed [rpm]": [1551, 1380],
            "Torque [Nm]": [42.8, 55.2],
            "Tool wear [min]": [5, 180],
        }
    )

    results = predict_failure_risk(trained_pipeline, df_sample)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert "machine_failure_prediction" in r
        assert "failure_probability" in r


def test_predict_failure_risk_list_of_dicts(trained_pipeline):
    """Verify batch predictions on a list of dictionaries."""
    records = [
        {
            "Type": "H",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10,
        }
    ]
    results = predict_failure_risk(trained_pipeline, records)
    assert isinstance(results, list)
    assert len(results) == 1


def test_predict_failure_risk_missing_features(trained_pipeline):
    """Verify ValueError is raised if required features are missing."""
    incomplete_sample = {
        "Type": "M",
        "Air temperature [K]": 298.1,
    }
    with pytest.raises(ValueError, match="missing required features"):
        predict_failure_risk(trained_pipeline, incomplete_sample)


def test_predict_failure_risk_none_model():
    """Verify ValueError is raised if model is None."""
    with pytest.raises(ValueError, match="Model object cannot be None"):
        predict_failure_risk(None, {"Type": "M"})


def test_predict_failure_risk_empty_data(trained_pipeline):
    """Verify ValueError is raised if data is empty."""
    with pytest.raises(ValueError, match="contains 0 records"):
        predict_failure_risk(trained_pipeline, pd.DataFrame())


def test_predict_failure_risk_ignores_leakage_columns(trained_pipeline):
    """Verify that extraneous leakage columns in input are safely dropped without error."""
    sample_with_leakage = {
        "UDI": 1,
        "Product ID": "M14860",
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 5,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0,
    }
    result = predict_failure_risk(trained_pipeline, sample_with_leakage)
    assert isinstance(result, dict)
    assert "machine_failure_prediction" in result
