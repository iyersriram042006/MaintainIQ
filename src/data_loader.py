"""Data loading and validation utilities for MaintainIQ.

Ensures the integrity of the UCI AI4I 2020 Predictive Maintenance Dataset.
"""

from pathlib import Path
from typing import Optional, Set
import pandas as pd

from config.settings import (
    ALL_EXPECTED_COLUMNS,
    CATEGORICAL_FEATURES,
    FAILURE_TYPE_COLUMNS,
    NUMERICAL_FEATURES,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)

# Valid categories for the 'Type' feature
VALID_PRODUCT_TYPES: Set[str] = {"L", "M", "H"}


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate that the raw AI4I dataset satisfies schema and value constraints.

    Checks performed:
    1. Dataset is not None and not empty.
    2. All expected columns exist (IDs, 6 input features, target, and 5 failure modes).
    3. Target column has no missing values and only contains binary values (0 or 1).
    4. Categorical feature 'Type' contains only valid variant types (L, M, H).
    5. All numerical features are numeric and have plausible physical values.
    6. Failure-mode indicator columns are binary.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty. Expected at least one row of data.")

    # 1. Column existence check
    missing_columns = [col for col in ALL_EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Expected all {len(ALL_EXPECTED_COLUMNS)} columns: {ALL_EXPECTED_COLUMNS}"
        )

    # 2. Target column validation
    target_series = df[TARGET_COLUMN]
    if target_series.isnull().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains null or missing values.")

    unique_targets = set(target_series.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must only contain binary values (0, 1). "
            f"Found invalid values: {unique_targets - {0, 1}}"
        )

    # 3. Categorical feature validation
    for cat_col in CATEGORICAL_FEATURES:
        unique_types = set(df[cat_col].dropna().unique())
        invalid_types = unique_types - VALID_PRODUCT_TYPES
        if invalid_types:
            raise ValueError(
                f"Categorical column '{cat_col}' contains invalid categories: {invalid_types}. "
                f"Expected subset of: {VALID_PRODUCT_TYPES}"
            )

    # 4. Numerical features validation
    for num_col in NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            raise ValueError(f"Numerical feature '{num_col}' must have a numeric dtype.")

    # 5. Failure-mode columns validation (must be binary)
    for fail_col in FAILURE_TYPE_COLUMNS:
        unique_fail_vals = set(df[fail_col].dropna().unique())
        if not unique_fail_vals.issubset({0, 1}):
            raise ValueError(
                f"Failure-mode column '{fail_col}' must be binary (0, 1). "
                f"Found invalid values: {unique_fail_vals - {0, 1}}"
            )


def load_raw_data(
    filepath: Optional[Path] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load and optionally validate the raw UCI AI4I 2020 dataset from disk.

    Args:
        filepath: Path to the CSV file. If None, defaults to RAW_DATA_FILE.
        validate: Whether to run validate_raw_data() on the loaded DataFrame.

    Returns:
        Validated DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file is empty or schema validation fails.
    """
    target_path = Path(filepath) if filepath else RAW_DATA_FILE

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{target_path}'. "
            "Please run scripts/download_data.py to download it."
        )

    if target_path.stat().st_size == 0:
        raise ValueError(f"Dataset file at '{target_path}' is empty (0 bytes).")

    # Use 'utf-8-sig' to automatically strip any UTF-8 Byte Order Mark (BOM)
    df = pd.read_csv(target_path, encoding="utf-8-sig")

    if validate:
        validate_raw_data(df)

    return df
