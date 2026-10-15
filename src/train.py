"""Model training, hyperparameter configuration, and artifact management for MaintainIQ.

Trains and benchmarks:
- Logistic Regression (class_weight='balanced')
- Random Forest (class_weight='balanced')
- XGBoost (scale_pos_weight derived from training set)
- CatBoost (auto_class_weights='Balanced')
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config.settings import (
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TABLES_DIR,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    build_comparison_dataframe,
    evaluate_model_pipeline,
    save_evaluation_reports,
)
from src.preprocessing import (
    build_preprocessor,
    clean_raw_data,
    create_stratified_split,
    split_features_target,
)

# Documentation of imbalance mitigation strategies per model
IMBALANCE_STRATEGIES: Dict[str, str] = {
    "logistic_regression": "class_weight='balanced' (inverse class frequency reweighting)",
    "random_forest": "class_weight='balanced' (subsample-balanced bootstrap trees)",
    "xgboost": "scale_pos_weight=n_neg/n_pos (computed strictly on training split)",
    "catboost": "auto_class_weights='Balanced' (CatBoost internal inverse frequency balancing)",
}

SUPPORTED_MODELS: List[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
]


def build_classifier_estimator(
    model_type: str,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a classifier with configured imbalance mitigation parameters.

    Args:
        model_type: Identifier string for the algorithm.
        y_train: Training labels used to compute class ratio for XGBoost.
        random_state: Seed for reproducibility.
        hyperparameters: Optional overrides for default classifier parameters.

    Returns:
        Instantiated classifier estimator.

    Raises:
        ValueError: If model_type is not supported.
    """
    model_key = model_type.lower()
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported options: {SUPPORTED_MODELS}"
        )

    params = hyperparameters.copy() if hyperparameters else {}

    # Calculate negative/positive ratio strictly on training labels for XGBoost
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    if model_key == "logistic_regression":
        default_params: Dict[str, Any] = {
            "class_weight": "balanced",
            "random_state": random_state,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    elif model_key == "random_forest":
        default_params = {
            "class_weight": "balanced",
            "random_state": random_state,
            "n_estimators": 100,
            "n_jobs": -1,
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    elif model_key == "xgboost":
        default_params = {
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_estimators": 100,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        default_params.update(params)
        return XGBClassifier(**default_params)

    elif model_key == "catboost":
        default_params = {
            "auto_class_weights": "Balanced",
            "random_seed": random_state,
            "iterations": 100,
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        default_params.update(params)
        return CatBoostClassifier(**default_params)


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    scale_numeric: bool = True,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train a leakage-safe scikit-learn Pipeline with preprocessing and classifier.

    Ensures that imputers, scalers, and the classifier are fit ONLY on X_train.

    Args:
        X_train: Training features DataFrame (6 approved features).
        y_train: Training target labels.
        model_type: One of 'logistic_regression', 'random_forest', 'xgboost', 'catboost'.
        scale_numeric: Whether to scale numeric features (recommended True for Logistic Regression).
        hyperparameters: Optional dictionary of model parameters.
        random_state: Seed for reproducibility.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor(scale_numeric=scale_numeric)
    estimator = build_classifier_estimator(
        model_type=model_type,
        y_train=y_train,
        random_state=random_state,
        hyperparameters=hyperparameters,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    # Fit pipeline strictly on training data
    pipeline.fit(X_train, y_train)
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Convenience wrapper conforming to the original train_model signature."""
    return train_classifier(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        hyperparameters=hyperparameters,
    )


def train_and_compare_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_to_train: Optional[List[str]] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, Any]]:
    """Train and evaluate all four classifiers on train and test splits.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        models_to_train: Optional subset of model names to run.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of:
        - Dictionary of fitted Pipeline objects keyed by model name.
        - Comparison DataFrame ranked by PR-AUC.
        - Comprehensive metrics dictionary for all models.
    """
    targets_to_run = models_to_train or SUPPORTED_MODELS
    fitted_pipelines: Dict[str, Pipeline] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    for model_name in targets_to_run:
        print(f"[MaintainIQ] Training {model_name}...")
        pipeline = train_classifier(
            X_train=X_train,
            y_train=y_train,
            model_type=model_name,
            scale_numeric=True,
            random_state=random_state,
        )
        fitted_pipelines[model_name] = pipeline

        # Evaluate strictly on test set
        eval_metrics = evaluate_model_pipeline(pipeline, X_test, y_test)
        eval_metrics["imbalance_strategy"] = IMBALANCE_STRATEGIES.get(
            model_name, "N/A"
        )
        metrics_by_model[model_name] = eval_metrics

    comparison_df = build_comparison_dataframe(metrics_by_model)
    return fitted_pipelines, comparison_df, metrics_by_model


def save_model(model: Any, filepath: Union[str, Path] = MODEL_PATH) -> Path:
    """Serialize a trained model or pipeline artifact to disk.

    Args:
        model: Trained model or pipeline object.
        filepath: Target destination path.

    Returns:
        Path to the saved artifact.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(filepath: Union[str, Path] = MODEL_PATH) -> Any:
    """Deserialize a trained model artifact from disk.

    Args:
        filepath: Path to the serialized model file.

    Returns:
        Deserialized model or pipeline object.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{target}'. "
            "Please run model training pipeline first."
        )
    return joblib.load(target)


def run_training_pipeline(
    data_path: Path = RAW_DATA_FILE,
    output_tables_dir: Path = TABLES_DIR,
    model_save_path: Path = MODEL_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute complete end-to-end training, benchmarking, and reporting workflow.

    Workflow:
    1. Loads raw dataset from data_path.
    2. Deduplicates and separates the 6 features from 'Machine failure'.
    3. Generates 80/20 stratified split.
    4. Trains Logistic Regression, Random Forest, XGBoost, and CatBoost.
    5. Computes Precision, Recall, F1, PR-AUC, ROC-AUC, and latency.
    6. Saves comparison table and JSON metrics to reports/tables/.
    7. Saves the best performing model (by PR-AUC) to models/model.joblib.

    Returns:
        Summary dictionary containing comparison table, metrics, and artifact paths.
    """
    print(f"[MaintainIQ] Loading raw data from: {data_path}")
    raw_df = load_raw_data(data_path, validate=True)
    cleaned_df = clean_raw_data(raw_df)
    X, y = split_features_target(cleaned_df)

    print(
        f"[MaintainIQ] Performing stratified split ({int((1-test_size)*100)}/{int(test_size*100)})..."
    )
    X_train, X_test, y_train, y_test = create_stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipelines, comparison_df, metrics_dict = train_and_compare_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
    )

    # Save reports
    saved_reports = save_evaluation_reports(
        comparison_df=comparison_df,
        full_metrics=metrics_dict,
        output_dir=output_tables_dir,
    )

    # Identify best model by PR-AUC (preferred metric for heavy imbalance)
    best_model_name = str(comparison_df.iloc[0]["Model"])
    best_pipeline = pipelines[best_model_name]
    best_model_path = save_model(best_pipeline, model_save_path)
    print(
        f"[MaintainIQ] Best model '{best_model_name}' saved to: {best_model_path}"
    )

    return {
        "comparison_df": comparison_df,
        "metrics": metrics_dict,
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "reports": saved_reports,
    }


if __name__ == "__main__":
    run_training_pipeline()
