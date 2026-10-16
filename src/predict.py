"""Inference and failure risk prediction utilities for MaintainIQ."""

from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from config.settings import EXCLUDED_FEATURES, FEATURE_COLUMNS, MODEL_PATH
from src.train import ensure_model_pipeline, load_model



def assess_risk_level(probability: float) -> str:
    """Classify failure risk probability into operational alert categories.

    Thresholds:
    - High Risk: >= 0.70 (urgent maintenance required)
    - Medium Risk: 0.30 to < 0.70 (schedule inspection soon)
    - Low Risk: < 0.30 (normal operating parameters)

    Args:
        probability: Float probability between 0.0 and 1.0.

    Returns:
        String indicator ('Low', 'Medium', 'High').
    """
    if probability >= 0.70:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    return "Low"


def format_single_prediction(
    prediction: int,
    probability: float,
) -> Dict[str, Any]:
    """Format single telemetry record prediction output."""
    return {
        "machine_failure_prediction": int(prediction),
        "failure_probability": float(probability),
        "risk_percentage": round(float(probability * 100.0), 2),
        "risk_level": assess_risk_level(probability),
    }


def predict_failure_risk(
    model: Any,
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate machine failure risk prediction and probabilities.

    Args:
        model: Trained scikit-learn Pipeline containing preprocessor and classifier.
        input_data: Features as a DataFrame, a single feature dictionary,
            or a list of feature dictionaries. Must contain the 6 approved features:
            ['Type', 'Air temperature [K]', 'Process temperature [K]',
             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'].

    Returns:
        Dictionary for single-item prediction or list of dictionaries for batch predictions.

    Raises:
        ValueError: If input is invalid or missing required features.
    """
    if model is None:
        raise ValueError("Model object cannot be None.")

    # Convert dictionary or list of dicts to DataFrame
    is_single_dict = False
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
        is_single_dict = True
    elif isinstance(input_data, list):
        if not input_data:
            raise ValueError("Input list of records cannot be empty.")
        df_input = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError(
            "input_data must be a pandas DataFrame, a dictionary, or a list of dictionaries."
        )

    if df_input.empty:
        raise ValueError("Input data contains 0 records.")

    # Validate presence of the 6 approved features
    missing_features = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required features: {missing_features}. "
            f"Expected exactly: {FEATURE_COLUMNS}"
        )

    # Strictly extract only the 6 approved features, discarding any extra columns
    X = df_input[FEATURE_COLUMNS].copy()

    # Generate predictions and probabilities
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds.astype(float)

    results = [
        format_single_prediction(p, prob)
        for p, prob in zip(preds, probs)
    ]

    return results[0] if is_single_dict else results


def validate_upload_schema(df: pd.DataFrame) -> bool:
    """Validate that input DataFrame contains all 6 required feature columns."""
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    expected_cols = set(FEATURE_COLUMNS)
    return expected_cols.issubset(set(df.columns))


def predict_batch_dataframe(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Run batch inference on a DataFrame and return a copy with prediction columns.

    Args:
        model: Trained pipeline.
        df: Input DataFrame containing the 6 required features.

    Returns:
        DataFrame with original columns plus:
        - 'predicted_failure' (0 or 1)
        - 'failure_probability' (0.0 to 1.0)
        - 'risk_percentage' (0.0% to 100.0%)
        - 'risk_level' ('Low', 'Medium', 'High')

    Raises:
        ValueError: If required feature columns are missing.
    """
    if not validate_upload_schema(df):
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        raise ValueError(f"Input DataFrame is missing required features: {missing}")

    predictions = predict_failure_risk(model, df)
    result_df = df.copy()
    result_df["predicted_failure"] = [p["machine_failure_prediction"] for p in predictions]
    result_df["failure_probability"] = [round(p["failure_probability"], 4) for p in predictions]
    result_df["risk_percentage"] = [p["risk_percentage"] for p in predictions]
    result_df["risk_level"] = [p["risk_level"] for p in predictions]
    return result_df

