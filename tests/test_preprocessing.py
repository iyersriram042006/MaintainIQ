"""Tests for data preprocessing, feature/target separation, and leakage-safe pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from config.settings import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessing import (
    build_preprocessor,
    clean_raw_data,
    create_stratified_split,
    fit_and_transform_pipeline,
    preprocess_and_split,
    remove_duplicates,
    split_features_target,
)


def create_sample_df(n_samples: int = 20, failure_rate: float = 0.2) -> pd.DataFrame:
    """Create a sample dataset with all 14 AI4I columns for testing."""
    np.random.seed(42)
    n_failures = int(n_samples * failure_rate)
    target = np.array([1] * n_failures + [0] * (n_samples - n_failures))
    np.random.shuffle(target)

    return pd.DataFrame(
        {
            "UDI": list(range(1, n_samples + 1)),
            "Product ID": [f"M1486{i}" for i in range(n_samples)],
            "Type": np.random.choice(["L", "M", "H"], size=n_samples),
            "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n_samples),
            "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n_samples),
            "Rotational speed [rpm]": np.random.randint(1200, 2000, size=n_samples),
            "Torque [Nm]": np.random.uniform(20.0, 60.0, size=n_samples),
            "Tool wear [min]": np.random.randint(0, 240, size=n_samples),
            "Machine failure": target,
            "TWF": [0] * n_samples,
            "HDF": [0] * n_samples,
            "PWF": target,
            "OSF": [0] * n_samples,
            "RNF": [0] * n_samples,
        }
    )


# 1. Duplicate Handling Tests
def test_remove_duplicates_exact():
    """Verify exact duplicate rows are removed."""
    df = create_sample_df(5)
    duplicated_df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    assert len(duplicated_df) == 7

    cleaned = remove_duplicates(duplicated_df)
    assert len(cleaned) == 5
    assert not cleaned.duplicated().any()


def test_remove_duplicates_invalid_input():
    """Verify passing non-DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="valid pandas DataFrame"):
        remove_duplicates(None)


def test_clean_raw_data_backward_compatibility():
    """Verify clean_raw_data functions as alias to remove_duplicates."""
    df = create_sample_df(3)
    duplicated_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(clean_raw_data(duplicated_df)) == 3


# 2. Feature / Target Separation & Leakage Prevention Tests
def test_split_features_target_success():
    """Verify correct separation of 6 input features and target."""
    df = create_sample_df(10)
    X, y = split_features_target(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 10
    assert len(y) == 10
    assert y.name == TARGET_COLUMN
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == 6


def test_split_features_target_excludes_leakage_columns():
    """Ensure UDI, Product ID, and failure modes are strictly excluded from X."""
    df = create_sample_df(10)
    X, y = split_features_target(df)

    for excluded in EXCLUDED_FEATURES:
        assert excluded not in X.columns, f"Leakage column '{excluded}' found in X!"


def test_split_features_target_missing_columns():
    """Verify ValueError is raised if any of the 6 features is missing."""
    df = create_sample_df(5)
    df_missing = df.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        split_features_target(df_missing)


def test_split_features_target_missing_target():
    """Verify ValueError is raised if target column is missing."""
    df = create_sample_df(5)
    df_no_target = df.drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="Missing required target column"):
        split_features_target(df_no_target)


def test_split_features_target_null_target():
    """Verify ValueError is raised if target contains nulls without fabricating."""
    df = create_sample_df(5)
    df.loc[0, TARGET_COLUMN] = None

    with pytest.raises(ValueError, match="Target labels must not be fabricated"):
        split_features_target(df)


# 3. Stratified Splitting Tests
def test_create_stratified_split_ratio_and_stratification():
    """Verify reproducible 80/20 split and class ratio preservation."""
    df = create_sample_df(100, failure_rate=0.1)  # 10 failures, 90 non-failures
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = create_stratified_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Stratification check: 10% failures in train (8/80) and test (2/20)
    assert y_train.sum() == 8
    assert y_test.sum() == 2
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_create_stratified_split_reproducibility():
    """Verify identical splits are produced with the same random_state."""
    df = create_sample_df(30)
    X, y = split_features_target(df)

    X_train_1, X_test_1, _, _ = create_stratified_split(X, y, random_state=42)
    X_train_2, X_test_2, _, _ = create_stratified_split(X, y, random_state=42)

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)


def test_create_stratified_split_validation_errors():
    """Verify validation when lengths mismatch or target has single class."""
    X = pd.DataFrame({"Air temperature [K]": [300.0, 301.0]})
    y_mismatch = pd.Series([0])
    with pytest.raises(ValueError, match="Length mismatch"):
        create_stratified_split(X, y_mismatch)

    y_single_class = pd.Series([0, 0])
    with pytest.raises(ValueError, match="at least two distinct target classes"):
        create_stratified_split(X, y_single_class)


# 4. Preprocessing Pipeline & Data Leakage Prevention Tests
def test_build_preprocessor_structure():
    """Verify pipeline transformers and feature outputs."""
    preprocessor = build_preprocessor(scale_numeric=True)
    assert isinstance(preprocessor, ColumnTransformer)

    # Check transformer names
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_pipeline_no_data_leakage():
    """Verify that scaler/imputer statistics are fit STRICTLY on training data."""
    df = create_sample_df(50)
    X, y = split_features_target(df)
    X_train, X_test, _, _ = create_stratified_split(X, y, test_size=0.2, random_state=42)

    # Inject extreme values into test set
    X_test_with_outlier = X_test.copy()
    X_test_with_outlier.loc[X_test.index[0], "Air temperature [K]"] = 99999.0

    preprocessor = build_preprocessor(scale_numeric=True)
    fit_and_transform_pipeline(preprocessor, X_train=X_train, X_test=X_test_with_outlier)

    # Verify scaler mean is computed ONLY on X_train (not affected by 99999.0)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train_air_temp_mean = X_train["Air temperature [K]"].mean()
    assert scaler.mean_[0] == pytest.approx(train_air_temp_mean, rel=1e-3)
    assert scaler.mean_[0] < 320.0  # Must not reflect outlier


def test_pipeline_handles_missing_values_safely():
    """Verify imputers handle missing values without errors using training stats."""
    df = create_sample_df(30)
    X, y = split_features_target(df)
    X_train, X_test, _, _ = create_stratified_split(X, y, test_size=0.2, random_state=42)

    # Introduce NaNs into features
    X_train_na = X_train.copy()
    X_train_na.loc[X_train.index[0], "Torque [Nm]"] = np.nan
    X_train_na.loc[X_train.index[1], "Type"] = np.nan

    X_test_na = X_test.copy()
    X_test_na.loc[X_test.index[0], "Torque [Nm]"] = np.nan

    preprocessor = build_preprocessor(scale_numeric=True)
    X_tr_trans, X_te_trans = fit_and_transform_pipeline(
        preprocessor, X_train=X_train_na, X_test=X_test_na
    )

    # Transformed arrays should contain zero NaNs
    assert not np.isnan(X_tr_trans).any()
    assert not np.isnan(X_te_trans).any()


def test_preprocess_and_split_end_to_end():
    """Verify complete end-to-end preprocessing function."""
    df = create_sample_df(100, failure_rate=0.1)
    result = preprocess_and_split(df, test_size=0.2, random_state=42, output_dataframe=True)

    assert "X_train" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_test" in result
    assert "X_train_transformed" in result
    assert "X_test_transformed" in result
    assert "preprocessor" in result
    assert "feature_names" in result

    # 5 numeric + 3 one-hot columns (Type_L, Type_M, Type_H) = 8 features
    assert len(result["feature_names"]) == 8
    assert result["X_train_transformed"].shape == (80, 8)
    assert result["X_test_transformed"].shape == (20, 8)
    assert isinstance(result["X_train_transformed"], pd.DataFrame)
