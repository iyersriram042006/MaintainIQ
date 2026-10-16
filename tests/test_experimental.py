"""Tests for optional syllabus-permitted experimental algorithms (K-Means and Linear Regression)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.experimental import cluster_operating_regimes, train_torque_regression


@pytest.fixture
def mock_numeric_data() -> pd.DataFrame:
    """Fixture providing continuous physical machine parameters."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n),
            "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n),
            "Rotational speed [rpm]": np.random.randint(1200, 2000, size=n),
            "Torque [Nm]": np.random.uniform(20.0, 60.0, size=n),
            "Tool wear [min]": np.random.randint(0, 240, size=n),
        }
    )


def test_cluster_operating_regimes(mock_numeric_data):
    """Verify K-Means operating regime clustering and profile summaries."""
    kmeans, clustered_df, summary = cluster_operating_regimes(
        mock_numeric_data, n_clusters=3, random_state=42
    )

    assert isinstance(kmeans, KMeans)
    assert "operating_regime_cluster" in clustered_df.columns
    assert len(clustered_df) == len(mock_numeric_data)
    assert set(clustered_df["operating_regime_cluster"].unique()).issubset({0, 1, 2})

    assert summary["n_clusters"] == 3
    assert "inertia" in summary
    assert len(summary["profiles"]) == 3
    for cid in range(3):
        assert "sample_count" in summary["profiles"][cid]
        assert "mean_torque" in summary["profiles"][cid]


def test_train_torque_regression(mock_numeric_data):
    """Verify Linear Regression for continuous cutting torque modeling."""
    pipeline, metrics = train_torque_regression(mock_numeric_data)

    assert isinstance(pipeline, Pipeline)
    assert "r2_score" in metrics
    assert "mae_nm" in metrics
    assert "rmse_nm" in metrics
    assert isinstance(metrics["r2_score"], float)
    assert metrics["mae_nm"] >= 0.0
    assert metrics["rmse_nm"] >= 0.0


def test_cluster_operating_regimes_invalid_inputs(mock_numeric_data):
    """Verify input validation for K-Means clustering."""
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        cluster_operating_regimes(None)

    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        cluster_operating_regimes(pd.DataFrame())

    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        cluster_operating_regimes(mock_numeric_data, n_clusters=0)

    with pytest.raises(ValueError, match="must be >= n_clusters"):
        cluster_operating_regimes(mock_numeric_data.head(2), n_clusters=5)


def test_train_torque_regression_invalid_inputs(mock_numeric_data):
    """Verify input validation for torque Linear Regression."""
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        train_torque_regression(None)

    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        train_torque_regression(pd.DataFrame())

    df_without_torque = mock_numeric_data.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Target 'Torque \\[Nm\\]' must be in X_features"):
        train_torque_regression(df_without_torque)

