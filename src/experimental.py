"""Optional experimental modules permitted by the syllabus.

Contains:
1. K-Means Clustering: Unsupervised segmentation of milling machine operating regimes.
2. Linear Regression: Numerical estimation of continuous machine operating parameters (e.g., cutting torque).

These algorithms are strictly separated from the primary binary machine failure classification task.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import NUMERICAL_FEATURES, RANDOM_STATE


def cluster_operating_regimes(
    X_num: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> Tuple[KMeans, pd.DataFrame, Dict[str, Any]]:
    """Segment operational telemetry into unsupervised machine working regimes via K-Means.

    Args:
        X_num: DataFrame containing numeric operational features.
        n_clusters: Number of operating clusters (default: 3 regimes: Normal, Heavy Load, Thermal Stress).
        random_state: Seed for reproducible centroid initialization.

    Returns:
        Tuple of (fitted KMeans model, DataFrame with cluster assignments, cluster summary statistics).
    """
    if X_num is None or not isinstance(X_num, pd.DataFrame) or X_num.empty:
        raise ValueError("Input X_num must be a non-empty pandas DataFrame.")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")

    # Retain only numerical features
    numeric_cols = [c for c in X_num.columns if c in NUMERICAL_FEATURES or pd.api.types.is_numeric_dtype(X_num[c])]
    if not numeric_cols:
        raise ValueError("No numerical columns found in X_num for clustering.")
    if len(X_num) < n_clusters:
        raise ValueError(f"Number of samples ({len(X_num)}) must be >= n_clusters ({n_clusters}).")
    X_numeric = X_num[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_clustered = X_num.copy()
    df_clustered["operating_regime_cluster"] = cluster_labels

    cluster_profiles: Dict[int, Dict[str, float]] = {}
    for cluster_id in range(n_clusters):
        subset = df_clustered[df_clustered["operating_regime_cluster"] == cluster_id]
        cluster_profiles[cluster_id] = {
            "sample_count": len(subset),
            "mean_torque": float(subset["Torque [Nm]"].mean()) if "Torque [Nm]" in subset else 0.0,
            "mean_speed": float(subset["Rotational speed [rpm]"].mean()) if "Rotational speed [rpm]" in subset else 0.0,
            "mean_tool_wear": float(subset["Tool wear [min]"].mean()) if "Tool wear [min]" in subset else 0.0,
            "mean_temp_delta": float((subset["Process temperature [K]"] - subset["Air temperature [K]"]).mean())
            if "Process temperature [K]" in subset and "Air temperature [K]" in subset
            else 0.0,
        }

    return kmeans, df_clustered, {
        "inertia": float(kmeans.inertia_),
        "n_clusters": n_clusters,
        "profiles": cluster_profiles,
    }


def train_torque_regression(
    X_features: pd.DataFrame,
    y_target: Optional[pd.Series] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train a Linear Regression model to predict continuous cutting Torque [Nm] from operating physics.

    Torque is mathematically related to spindle speed and power. This regression experiment
    models continuous physical dependencies rather than failure classification.

    Args:
        X_features: Feature DataFrame. If y_target is None, expects 'Torque [Nm]' column in X_features.
        y_target: Optional target Series. If None, extracted from X_features['Torque [Nm]'].

    Returns:
        Tuple of (fitted Pipeline, dictionary of regression metrics: R2, RMSE, MAE).
    """
    if X_features is None or not isinstance(X_features, pd.DataFrame) or X_features.empty:
        raise ValueError("Input X_features must be a non-empty pandas DataFrame.")

    if y_target is None:
        if "Torque [Nm]" not in X_features.columns:
            raise ValueError("Target 'Torque [Nm]' must be in X_features if y_target is None.")
        y = X_features["Torque [Nm]"].copy()
        X = X_features.drop(columns=["Torque [Nm]"])
    else:
        y = y_target.copy()
        X = X_features.copy()

    # Retain only numerical features present in X
    numeric_cols = [c for c in X.columns if c in NUMERICAL_FEATURES or pd.api.types.is_numeric_dtype(X[c])]
    if not numeric_cols:
        raise ValueError("No numerical predictor features available for regression.")
    X_num = X[numeric_cols]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])

    pipeline.fit(X_num, y)
    preds = pipeline.predict(X_num)

    metrics = {
        "r2_score": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse_nm": float(np.sqrt(mean_squared_error(y, preds))),
        "mae_nm": float(mean_absolute_error(y, preds)),
    }

    return pipeline, metrics
