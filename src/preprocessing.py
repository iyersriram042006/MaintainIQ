"""Data preprocessing, splitting, and pipeline engineering for MaintainIQ.

Implements strict target-leakage prevention and reproducible stratified splitting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

# Valid categories for product Type
PRODUCT_TYPE_CATEGORIES: List[str] = ["L", "M", "H"]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with exact duplicates removed.

    Raises:
        ValueError: If df is None or not a DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")
    return df.drop_duplicates().copy()


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for remove_duplicates to maintain backward compatibility."""
    return remove_duplicates(df)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the six approved feature columns from the binary target column.

    Enforces strict prevention of data leakage:
    - Never includes identifiers ('UDI', 'Product ID').
    - Never includes failure-mode labels ('TWF', 'HDF', 'PWF', 'OSF', 'RNF').

    Args:
        df: Input DataFrame containing features and target.

    Returns:
        Tuple of (X feature DataFrame, y target Series).

    Raises:
        ValueError: If required features or target column are missing,
            or if target contains missing values.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    # Validate target presence and values
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required target column '{TARGET_COLUMN}'. "
            "Target labels must not be fabricated."
        )

    y = df[TARGET_COLUMN].copy()
    if y.isnull().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains null/missing values. "
            "Target labels must not be fabricated."
        )

    # Validate feature presence
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}. "
            f"Expected exactly these 6 features: {FEATURE_COLUMNS}"
        )

    # Strictly extract only the 6 approved features
    X = df[FEATURE_COLUMNS].copy()

    # Verify no leakage columns leaked into X
    leaked_cols = [col for col in EXCLUDED_FEATURES if col in X.columns]
    if leaked_cols:
        raise ValueError(
            f"Data leakage detected! Excluded columns found in feature set X: {leaked_cols}"
        )

    return X, y


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a reproducible stratified train/test split.

    Preserves the binary failure class ratio in both train and test partitions
    without performing resampling.

    Args:
        X: Feature DataFrame (must contain only the approved features).
        y: Target label Series.
        test_size: Proportion of the dataset to allocate to the test split (default: 0.2).
        random_state: Seed for reproducible random splitting (default: 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If inputs have mismatched lengths, insufficient classes,
            or invalid split proportions.
    """
    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows while y has {len(y)} labels."
        )

    unique_classes = np.unique(y.dropna())
    if len(unique_classes) < 2:
        raise ValueError(
            "Stratified split requires at least two distinct target classes."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    scale_numeric: bool = True,
    output_dataframe: bool = False,
) -> ColumnTransformer:
    """Build an unfitted scikit-learn preprocessing ColumnTransformer.

    Architecture:
    - Categorical features ('Type'):
      1. SimpleImputer(strategy='most_frequent')
      2. OneHotEncoder(categories=[['L', 'M', 'H']], handle_unknown='ignore', sparse_output=False)
    - Numerical features (Air temp, Process temp, Speed, Torque, Tool wear):
      1. SimpleImputer(strategy='median')
      2. StandardScaler() (if scale_numeric=True) or passthrough (if scale_numeric=False)
    - Remainder: 'drop' to discard any unauthorized columns passed at runtime.

    Args:
        scale_numeric: If True, applies StandardScaler to numerical features.
            If False, only imputes missing values and passes numerical features through.
        output_dataframe: If True, configures transformer to output a pandas DataFrame.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    # Numerical sub-pipeline
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical sub-pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[PRODUCT_TYPE_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if output_dataframe:
        preprocessor.set_output(transform="pandas")

    return preprocessor


def fit_and_transform_pipeline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]]]:
    """Fit preprocessor strictly on X_train and transform X_train and optional X_test.

    Prevents data leakage by ensuring the preprocessor is NEVER fit on X_test.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training feature DataFrame used to learn imputation and scaling stats.
        X_test: Optional test feature DataFrame to transform using training statistics.

    Returns:
        Tuple of (X_train_transformed, X_test_transformed). If X_test is None,
        the second element is None.
    """
    # 1. Fit ONLY on training data
    preprocessor.fit(X_train)

    # 2. Transform train
    X_train_transformed = preprocessor.transform(X_train)

    # 3. Transform test if provided (without fitting!)
    X_test_transformed = preprocessor.transform(X_test) if X_test is not None else None

    return X_train_transformed, X_test_transformed


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    scale_numeric: bool = True,
    output_dataframe: bool = False,
) -> Dict[str, Any]:
    """Execute complete preprocessing and splitting pipeline.

    Workflow:
    1. Deduplicates raw DataFrame.
    2. Separates 6 approved features from 'Machine failure' target.
    3. Stratifies 80/20 train/test split.
    4. Builds and fits preprocessor ONLY on training features.
    5. Transforms both training and test features.

    Args:
        df: Raw input DataFrame.
        test_size: Proportion of test split (default: 0.2).
        random_state: Random state for reproducibility (default: 42).
        scale_numeric: Whether to scale numeric features (default: True).
        output_dataframe: Whether transformed features should be DataFrames (default: False).

    Returns:
        Dictionary containing:
        - 'X_train': raw training feature split (pd.DataFrame)
        - 'X_test': raw test feature split (pd.DataFrame)
        - 'y_train': training labels (pd.Series)
        - 'y_test': test labels (pd.Series)
        - 'X_train_transformed': transformed training features
        - 'X_test_transformed': transformed test features
        - 'preprocessor': fitted ColumnTransformer instance
        - 'feature_names': list of transformed feature names
    """
    cleaned_df = clean_raw_data(df)
    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, y_test = create_stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric,
        output_dataframe=output_dataframe,
    )

    X_train_trans, X_test_trans = fit_and_transform_pipeline(
        preprocessor,
        X_train=X_train,
        X_test=X_test,
    )

    feature_names = (
        list(preprocessor.get_feature_names_out())
        if hasattr(preprocessor, "get_feature_names_out")
        else []
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_transformed": X_train_trans,
        "X_test_transformed": X_test_trans,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
