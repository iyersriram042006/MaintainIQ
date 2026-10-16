"""Model training, hyperparameter configuration, and artifact management for MaintainIQ.

Trains and benchmarks the four approved classification algorithms:
- Logistic Regression (class_weight='balanced')
- Decision Tree Classifier (class_weight='balanced', max_depth=8, min_samples_leaf=4)
- K-Nearest Neighbors Classifier (weights='distance', n_neighbors=7)
- Support Vector Machine (SVC, probability=True, class_weight='balanced')
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    "decision_tree": "class_weight='balanced' (cost-sensitive tree split reweighting)",
    "knn": "distance-weighted neighbors (weights='distance', k=7)",
    "svm": "class_weight='balanced', probability=True (margin-weighted RBF support vectors)",
}

SUPPORTED_MODELS: List[str] = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "svm",
]


def build_classifier_estimator(
    model_type: str,
    y_train: Optional[pd.Series] = None,
    random_state: int = RANDOM_STATE,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a classifier with configured imbalance mitigation parameters.

    Args:
        model_type: Identifier string for the algorithm ('logistic_regression',
            'decision_tree', 'knn', 'svm').
        y_train: Optional training labels (retained for backward interface compatibility).
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

    if model_key == "logistic_regression":
        default_params: Dict[str, Any] = {
            "class_weight": "balanced",
            "random_state": random_state,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    elif model_key == "decision_tree":
        default_params = {
            "class_weight": "balanced",
            "max_depth": 8,
            "min_samples_leaf": 4,
            "random_state": random_state,
        }
        default_params.update(params)
        return DecisionTreeClassifier(**default_params)

    elif model_key == "knn":
        default_params = {
            "n_neighbors": 7,
            "weights": "distance",
        }
        default_params.update(params)
        return KNeighborsClassifier(**default_params)

    elif model_key == "svm":
        default_params = {
            "class_weight": "balanced",
            "probability": True,
            "kernel": "rbf",
            "random_state": random_state,
        }
        default_params.update(params)
        return SVC(**default_params)


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
        model_type: One of 'logistic_regression', 'decision_tree', 'knn', 'svm'.
        scale_numeric: Whether to scale numeric features.
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
    model_type: str = "decision_tree",
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
    """Train and evaluate the four approved classifiers on train and test splits.

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
        t0 = time.perf_counter()
        pipeline = train_classifier(
            X_train=X_train,
            y_train=y_train,
            model_type=model_name,
            scale_numeric=True,
            random_state=random_state,
        )
        train_time_sec = time.perf_counter() - t0
        fitted_pipelines[model_name] = pipeline

        # Evaluate strictly on test set
        eval_metrics = evaluate_model_pipeline(pipeline, X_test, y_test)
        eval_metrics["imbalance_strategy"] = IMBALANCE_STRATEGIES.get(
            model_name, "N/A"
        )
        eval_metrics["training_time_sec"] = float(train_time_sec)
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


def train_champion_model(
    data_path: Path = RAW_DATA_FILE,
    model_save_path: Optional[Union[str, Path]] = MODEL_PATH,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> Pipeline:
    """Train exclusively the champion Decision Tree pipeline on the training split.

    Used for fast initial setup or on-demand provisioning without retraining
    all four benchmark models.

    Args:
        data_path: Path to raw dataset CSV.
        model_save_path: Optional path to serialize the fitted model.
        random_state: Seed for reproducible stratified split and model fit.
        test_size: Proportion of dataset for held-out evaluation.

    Returns:
        Fitted scikit-learn Pipeline (preprocessing + Decision Tree).
    """
    raw_df = load_raw_data(data_path, validate=True)
    cleaned_df = clean_raw_data(raw_df)
    X, y = split_features_target(cleaned_df)
    X_train, _, y_train, _ = create_stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = train_classifier(
        X_train=X_train,
        y_train=y_train,
        model_type="decision_tree",
        scale_numeric=True,
        random_state=random_state,
    )

    if model_save_path:
        save_model(pipeline, model_save_path)

    return pipeline


def ensure_model_pipeline(
    model_path: Union[str, Path] = MODEL_PATH,
    data_path: Path = RAW_DATA_FILE,
    auto_download: bool = True,
) -> Any:
    """Ensure a trained champion model pipeline is available.

    1. If model_path exists on disk, deserializes and returns it immediately.
    2. If missing, ensures the raw dataset is present (downloading from UCI if needed).
    3. Trains exclusively the champion Decision Tree pipeline, saves it to model_path,
       and returns the fitted pipeline.

    Args:
        model_path: Target path to the serialized model file.
        data_path: Path to the raw CSV dataset.
        auto_download: Whether to attempt automatic dataset download if missing.

    Returns:
        Trained scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If model and dataset are both missing and auto_download is False.
    """
    target_model = Path(model_path)
    if target_model.exists():
        return load_model(target_model)

    target_data = Path(data_path)
    if not target_data.exists():
        if auto_download:
            from scripts.download_data import download_dataset
            print(f"[MaintainIQ] Dataset missing at '{target_data}'. Downloading from UCI...")
            download_dataset(target_file=target_data)
        else:
            raise FileNotFoundError(
                f"Model missing at '{target_model}' and dataset missing at '{target_data}'."
            )

    print(f"[MaintainIQ] Provisioning champion Decision Tree pipeline to '{target_model}'...")
    return train_champion_model(data_path=target_data, model_save_path=target_model)


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
    4. Trains Logistic Regression, Decision Tree, KNN, and SVM.
    5. Computes Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, latency, and training time.
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
