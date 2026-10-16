"""Tests for model explainability, permutation importance, TreeSHAP, and reporting."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config.settings import FEATURE_COLUMNS, RANDOM_STATE
from src.explain import (
    CAUSATION_DISCLAIMER,
    compute_permutation_importance,
    explain_prediction,
    plot_permutation_importance,
    save_feature_importance_reports,
)
from src.train import train_classifier


@pytest.fixture
def trained_tree_pipeline() -> Pipeline:
    """Fixture providing a small trained Decision Tree pipeline."""
    np.random.seed(42)
    n = 40
    X_train = pd.DataFrame(
        {
            "Type": np.random.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n),
            "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n),
            "Rotational speed [rpm]": np.random.randint(1200, 2000, size=n),
            "Torque [Nm]": np.random.uniform(20.0, 60.0, size=n),
            "Tool wear [min]": np.random.randint(0, 240, size=n),
        }
    )
    y_train = pd.Series([1] * 4 + [0] * 36, name="Machine failure")
    return train_classifier(X_train, y_train, model_type="decision_tree")


@pytest.fixture
def mock_eval_data() -> tuple:
    """Fixture providing evaluation data."""
    np.random.seed(123)
    n = 20
    X_eval = pd.DataFrame(
        {
            "Type": np.random.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n),
            "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n),
            "Rotational speed [rpm]": np.random.randint(1200, 2000, size=n),
            "Torque [Nm]": np.random.uniform(20.0, 60.0, size=n),
            "Tool wear [min]": np.random.randint(0, 240, size=n),
        }
    )
    y_eval = pd.Series([1] * 3 + [0] * 17, name="Machine failure")
    return X_eval, y_eval


def test_compute_permutation_importance_transformed_features(
    trained_tree_pipeline, mock_eval_data
):
    """Verify output shape, feature names, and sorting for transformed features."""
    X_eval, y_eval = mock_eval_data
    df_imp = compute_permutation_importance(
        pipeline=trained_tree_pipeline,
        X_eval=X_eval,
        y_eval=y_eval,
        n_repeats=5,
        random_state=42,
        use_transformed_features=True,
    )

    assert isinstance(df_imp, pd.DataFrame)
    assert list(df_imp.columns) == ["feature", "importance_mean", "importance_std"]
    # 5 numeric + 3 one-hot columns = 8 features
    assert len(df_imp) == 8

    # Verify expected feature names
    expected_names = {
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_L",
        "Type_M",
        "Type_H",
    }
    assert set(df_imp["feature"]) == expected_names

    # Verify descending sort
    means = df_imp["importance_mean"].tolist()
    assert means == sorted(means, reverse=True)


def test_compute_permutation_importance_raw_features(
    trained_tree_pipeline, mock_eval_data
):
    """Verify output shape and names when evaluating on the 6 raw inputs."""
    X_eval, y_eval = mock_eval_data
    df_imp = compute_permutation_importance(
        pipeline=trained_tree_pipeline,
        X_eval=X_eval,
        y_eval=y_eval,
        n_repeats=5,
        random_state=42,
        use_transformed_features=False,
    )

    assert len(df_imp) == 6
    assert set(df_imp["feature"]) == set(FEATURE_COLUMNS)


def test_compute_permutation_importance_deterministic(
    trained_tree_pipeline, mock_eval_data
):
    """Verify deterministic outputs with fixed random_state."""
    X_eval, y_eval = mock_eval_data
    df_imp1 = compute_permutation_importance(
        trained_tree_pipeline, X_eval, y_eval, n_repeats=5, random_state=42
    )
    df_imp2 = compute_permutation_importance(
        trained_tree_pipeline, X_eval, y_eval, n_repeats=5, random_state=42
    )

    pd.testing.assert_frame_equal(df_imp1, df_imp2)


def test_compute_permutation_importance_errors(trained_tree_pipeline):
    """Verify error handling on invalid inputs."""
    X = pd.DataFrame({"Type": ["M"]})
    y = pd.Series([0])

    with pytest.raises(ValueError, match="Pipeline cannot be None"):
        compute_permutation_importance(None, X, y)

    with pytest.raises(ValueError, match="matching lengths"):
        compute_permutation_importance(trained_tree_pipeline, X, pd.Series([0, 1]))

    with pytest.raises(ValueError, match="cannot be empty"):
        compute_permutation_importance(
            trained_tree_pipeline, pd.DataFrame(), pd.Series([], dtype=int)
        )


def test_explain_prediction_structure_and_disclaimer(trained_tree_pipeline):
    """Verify explain_prediction returns attributions, method, and causation disclaimer."""
    sample = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 5,
    }

    explanation = explain_prediction(trained_tree_pipeline, sample)
    assert isinstance(explanation, dict)
    assert "prediction" in explanation
    assert "failure_probability" in explanation
    assert "feature_attributions" in explanation
    assert "method" in explanation
    assert "disclaimer" in explanation
    assert "causation" in explanation["disclaimer"].lower()

    # Verify all 8 transformed features are present in attributions
    attributions = explanation["feature_attributions"]
    assert len(attributions) == 8
    assert "Torque [Nm]" in attributions
    assert "Tool wear [min]" in attributions


def test_explain_prediction_dataframe_input(trained_tree_pipeline):
    """Verify explain_prediction works on 1-row DataFrame and rejects multi-row."""
    df_single = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 5,
            }
        ]
    )

    exp = explain_prediction(trained_tree_pipeline, df_single)
    assert exp["prediction"] in [0, 1]

    df_multi = pd.concat([df_single, df_single], ignore_index=True)
    with pytest.raises(ValueError, match="single telemetry row"):
        explain_prediction(trained_tree_pipeline, df_multi)


def test_plot_permutation_importance(tmp_path: Path):
    """Verify figure creation and image file integrity."""
    df_imp = pd.DataFrame(
        {
            "feature": ["Torque [Nm]", "Tool wear [min]", "Air temperature [K]"],
            "importance_mean": [0.55, 0.35, 0.15],
            "importance_std": [0.03, 0.02, 0.01],
        }
    )

    out_file = tmp_path / "test_importance.png"
    result_path = plot_permutation_importance(df_imp, out_file)

    assert result_path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 1000  # Non-empty PNG image


def test_save_feature_importance_reports(tmp_path: Path):
    """Verify saving CSV, Markdown, and PNG reports."""
    df_imp = pd.DataFrame(
        {
            "feature": ["Torque [Nm]", "Tool wear [min]"],
            "importance_mean": [0.55, 0.35],
            "importance_std": [0.03, 0.02],
        }
    )

    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"

    saved = save_feature_importance_reports(df_imp, tables_dir, figures_dir)

    assert saved["csv"].exists()
    assert saved["markdown"].exists()
    assert saved["figure"].exists()

    df_csv = pd.read_csv(saved["csv"])
    assert len(df_csv) == 2
    assert list(df_csv.columns) == ["feature", "importance_mean", "importance_std"]
