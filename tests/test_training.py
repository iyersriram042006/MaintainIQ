"""Tests for model training pipeline, imbalance mitigation, and artifact serialization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config.settings import FEATURE_COLUMNS, RANDOM_STATE
from src.train import (
    IMBALANCE_STRATEGIES,
    SUPPORTED_MODELS,
    build_classifier_estimator,
    ensure_model_pipeline,
    load_model,
    save_model,
    train_and_compare_all_models,
    train_champion_model,
    train_classifier,
    train_model,
)



def create_mock_train_data(n_samples: int = 40) -> tuple:
    """Create a small mock feature set and imbalanced target for training tests."""
    np.random.seed(42)
    n_failures = max(int(n_samples * 0.1), 1)
    y = pd.Series([1] * n_failures + [0] * (n_samples - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "Type": np.random.choice(["L", "M", "H"], size=n_samples),
            "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n_samples),
            "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n_samples),
            "Rotational speed [rpm]": np.random.randint(1200, 2000, size=n_samples),
            "Torque [Nm]": np.random.uniform(20.0, 60.0, size=n_samples),
            "Tool wear [min]": np.random.randint(0, 240, size=n_samples),
        }
    )
    return X, y



@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_train_classifier_all_supported_models(model_name: str):
    """Verify that all four classifiers train successfully within a Pipeline."""
    X_train, y_train = create_mock_train_data(30)
    pipeline = train_classifier(
        X_train=X_train,
        y_train=y_train,
        model_type=model_name,
        scale_numeric=True,
        random_state=RANDOM_STATE,
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # Test inference output
    preds = pipeline.predict(X_train)
    assert len(preds) == len(X_train)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test probability output
    probs = pipeline.predict_proba(X_train)
    assert probs.shape == (len(X_train), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_train_classifier_unsupported_model_raises_error():
    """Verify that unsupported model types raise ValueError."""
    X_train, y_train = create_mock_train_data(20)
    with pytest.raises(ValueError, match="Unsupported model_type"):
        train_classifier(X_train, y_train, model_type="unsupported_dnn")


def test_imbalance_strategies_documented():
    """Verify all 4 supported models have documented imbalance strategies."""
    for model_name in SUPPORTED_MODELS:
        assert model_name in IMBALANCE_STRATEGIES
        assert len(IMBALANCE_STRATEGIES[model_name]) > 10


def test_train_model_convenience_alias():
    """Verify train_model signature alias works identically."""
    X_train, y_train = create_mock_train_data(20)
    pipeline = train_model(X_train, y_train, model_type="logistic_regression")
    assert isinstance(pipeline, Pipeline)


def test_save_and_load_model_artifact(tmp_path: Path):
    """Verify model pipeline serialization and deserialization."""
    X_train, y_train = create_mock_train_data(25)
    pipeline = train_classifier(X_train, y_train, model_type="decision_tree")

    artifact_file = tmp_path / "test_model.joblib"
    saved_path = save_model(pipeline, artifact_file)
    assert saved_path == artifact_file
    assert artifact_file.exists()

    loaded_pipeline = load_model(artifact_file)
    original_preds = pipeline.predict(X_train)
    loaded_preds = loaded_pipeline.predict(X_train)
    np.testing.assert_array_equal(original_preds, loaded_preds)


def test_load_model_missing_raises_error(tmp_path: Path):
    """Verify loading non-existent model artifact raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(missing_file)


def test_train_and_compare_all_models_workflow():
    """Verify training and benchmarking of all four models together."""
    X, y = create_mock_train_data(40)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y.iloc[:30], y.iloc[30:]

    pipelines, comp_df, metrics = train_and_compare_all_models(
        X_train, y_train, X_test, y_test, random_state=42
    )

    assert len(pipelines) == 4
    for model_name in SUPPORTED_MODELS:
        assert model_name in pipelines
        assert model_name in metrics

    assert len(comp_df) == 4
    assert "PR-AUC" in comp_df.columns
    assert "F1-Score" in comp_df.columns
    assert "Model" in comp_df.columns


def create_mock_dataset_csv(filepath: Path, num_rows: int = 30) -> Path:
    """Helper to write a valid minimal raw dataset CSV for pipeline training tests."""
    df = pd.DataFrame(
        {
            "UDI": list(range(1, num_rows + 1)),
            "Product ID": [f"M1486{i}" for i in range(num_rows)],
            "Type": ["M"] * num_rows,
            "Air temperature [K]": [298.1 + i * 0.1 for i in range(num_rows)],
            "Process temperature [K]": [308.6 + i * 0.1 for i in range(num_rows)],
            "Rotational speed [rpm]": [1550 + i * 10 for i in range(num_rows)],
            "Torque [Nm]": [42.8 - i * 0.2 for i in range(num_rows)],
            "Tool wear [min]": [i * 5 for i in range(num_rows)],
            "Machine failure": [1 if i < 3 else 0 for i in range(num_rows)],
            "TWF": [0] * num_rows,
            "HDF": [0] * num_rows,
            "PWF": [1 if i < 3 else 0 for i in range(num_rows)],
            "OSF": [0] * num_rows,
            "RNF": [0] * num_rows,
        }
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return filepath


def test_ensure_model_pipeline_loads_existing_artifact(tmp_path: Path):
    """Verify that existing model artifact is loaded directly without data access."""
    X_train, y_train = create_mock_train_data(20)
    pipeline = train_classifier(X_train, y_train, model_type="decision_tree")
    model_file = tmp_path / "existing_model.joblib"
    save_model(pipeline, model_file)

    # Point data_path to non-existent file; it should NOT be touched
    missing_data_file = tmp_path / "does_not_exist.csv"
    loaded = ensure_model_pipeline(model_path=model_file, data_path=missing_data_file)
    assert isinstance(loaded, Pipeline)
    assert "classifier" in loaded.named_steps


def test_ensure_model_pipeline_provisions_when_model_missing(tmp_path: Path):
    """Verify champion model is auto-trained and saved when model artifact is missing."""
    data_file = create_mock_dataset_csv(tmp_path / "data" / "ai4i2020.csv", num_rows=25)
    model_file = tmp_path / "models" / "model.joblib"
    assert not model_file.exists()

    pipeline = ensure_model_pipeline(
        model_path=model_file, data_path=data_file, auto_download=False
    )
    assert model_file.exists()
    assert isinstance(pipeline, Pipeline)
    assert type(pipeline.named_steps["classifier"]).__name__ == "DecisionTreeClassifier"

    # Verify predictions work on sample record
    test_record = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1550,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 10,
            }
        ]
    )
    prob = pipeline.predict_proba(test_record)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_ensure_model_pipeline_missing_data_no_download_raises_error(tmp_path: Path):
    """Verify FileNotFoundError is raised if model and data are missing with auto_download=False."""
    missing_model = tmp_path / "models" / "missing.joblib"
    missing_data = tmp_path / "data" / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Model missing"):
        ensure_model_pipeline(
            model_path=missing_model, data_path=missing_data, auto_download=False
        )


def test_train_champion_model_only_trains_decision_tree(tmp_path: Path):
    """Verify train_champion_model trains exclusively the Decision Tree champion pipeline."""
    data_file = create_mock_dataset_csv(tmp_path / "raw.csv", num_rows=25)
    save_file = tmp_path / "champion.joblib"

    pipeline = train_champion_model(data_path=data_file, model_save_path=save_file)
    assert save_file.exists()
    assert isinstance(pipeline, Pipeline)
    assert type(pipeline.named_steps["classifier"]).__name__ == "DecisionTreeClassifier"
