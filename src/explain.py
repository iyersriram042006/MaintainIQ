"""Model explainability and feature attribution module for MaintainIQ.

Implements:
1. Permutation Feature Importance: Empirical global reliance across transformed features.
2. TreeSHAP Local Attributions: Instance-level feature contributions.
3. Publication-quality visualization and report persistence.

Scientific Disclaimer:
Feature importance and attribution metrics quantify empirical model reliance and
statistical association. They do not demonstrate physical causation or guarantee
underlying mechanical causality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config.settings import (
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TABLES_DIR,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.evaluate import dataframe_to_markdown
from src.preprocessing import clean_raw_data, create_stratified_split, split_features_target
from src.train import load_model

CAUSATION_DISCLAIMER: str = (
    "Note: Feature importance and SHAP attributions reflect empirical model reliance "
    "and statistical association on the dataset, not physical or mechanical causation."
)


def compute_permutation_importance(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    use_transformed_features: bool = True,
) -> pd.DataFrame:
    """Compute permutation feature importance on evaluation data.

    Measures the drop in the target evaluation metric (default: PR-AUC) when
    individual feature columns are randomly permuted.

    Args:
        pipeline: Trained scikit-learn Pipeline with 'preprocessor' and 'classifier'.
        X_eval: Evaluation features DataFrame.
        y_eval: Evaluation ground-truth labels.
        scoring: Metric to evaluate (default: 'average_precision' / PR-AUC).
        n_repeats: Number of permutation iterations per feature.
        random_state: Random state for deterministic permutation.
        use_transformed_features: If True, measures importance of the 8 transformed
            features (5 numerical + 3 one-hot Type columns) directly from the classifier.
            If False, measures importance on the 6 raw input features.

    Returns:
        DataFrame containing 'feature', 'importance_mean', and 'importance_std'
        ranked in descending order of importance_mean.

    Raises:
        ValueError: If pipeline is missing required steps or input is invalid.
    """
    if pipeline is None:
        raise ValueError("Pipeline cannot be None.")
    if len(X_eval) != len(y_eval):
        raise ValueError("X_eval and y_eval must have matching lengths.")
    if len(X_eval) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if use_transformed_features:
        if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain 'preprocessor' and 'classifier' steps."
            )

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        # Transform features using training-fitted preprocessor
        X_trans = preprocessor.transform(X_eval)
        feature_names = list(preprocessor.get_feature_names_out())

        result = permutation_importance(
            estimator=classifier,
            X=X_trans,
            y=y_eval,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        feature_names = list(X_eval.columns)
        result = permutation_importance(
            estimator=pipeline,
            X=X_eval,
            y=y_eval,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": [float(m) for m in result.importances_mean],
            "importance_std": [float(s) for s in result.importances_std],
        }
    )

    importance_df = (
        importance_df.sort_values(by="importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def explain_prediction(
    pipeline: Pipeline,
    instance: Union[pd.DataFrame, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute local feature attributions for a single machine telemetry record.

    Uses TreeSHAP for tree-based estimators (XGBoost, Random Forest, CatBoost),
    falling back to model coefficients or baseline deviation if SHAP is inapplicable.

    Args:
        pipeline: Trained scikit-learn Pipeline.
        instance: Single telemetry record as a dict or 1-row DataFrame.

    Returns:
        Dictionary containing predicted risk, base value, and feature attributions
        mapped to the transformed feature names.

    Raises:
        ValueError: If input is invalid or pipeline steps are missing.
    """
    if pipeline is None:
        raise ValueError("Pipeline cannot be None.")

    if isinstance(instance, dict):
        df_instance = pd.DataFrame([instance])
    elif isinstance(instance, pd.DataFrame):
        if len(instance) != 1:
            raise ValueError("explain_prediction expects a single telemetry row (1 record).")
        df_instance = instance.copy()
    else:
        raise ValueError("instance must be a dictionary or a 1-row pandas DataFrame.")

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if preprocessor is None or classifier is None:
        raise ValueError("Pipeline must contain 'preprocessor' and 'classifier'.")

    # Transform instance
    X_trans = preprocessor.transform(df_instance)
    feature_names = list(preprocessor.get_feature_names_out())

    # Predict risk probability
    pred = int(pipeline.predict(df_instance)[0])
    prob = float(
        pipeline.predict_proba(df_instance)[0, 1]
        if hasattr(pipeline, "predict_proba")
        else float(pred)
    )

    # Compute SHAP values via TreeExplainer if tree-based
    method_name = "TreeSHAP"
    base_val = 0.5
    attributions: Dict[str, float] = {}

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_trans)

        # Handle binary classification outputs (some versions return list of [neg, pos])
        if isinstance(shap_vals, list):
            target_shap = shap_vals[1]
        else:
            target_shap = shap_vals

        if hasattr(explainer, "expected_value"):
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                base_val = float(exp_val[1])
            else:
                base_val = float(exp_val)

        # Map SHAP attributions to feature names
        vals = target_shap[0]
        for name, val in zip(feature_names, vals):
            attributions[name] = float(val)

    except Exception as e:
        # Fallback for linear models or unexpected SHAP interface mismatches
        method_name = f"Coefficients/Fallback ({e.__class__.__name__})"
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
            for name, coef, x_val in zip(feature_names, coefs, X_trans[0]):
                attributions[name] = float(coef * x_val)
        else:
            for name, x_val in zip(feature_names, X_trans[0]):
                attributions[name] = float(x_val)

    # Sort attributions by absolute impact descending
    sorted_attributions = dict(
        sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    return {
        "prediction": pred,
        "failure_probability": prob,
        "risk_percentage": round(prob * 100.0, 2),
        "base_value": base_val,
        "feature_attributions": sorted_attributions,
        "method": method_name,
        "disclaimer": CAUSATION_DISCLAIMER,
    }


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "MaintainIQ — Permutation Feature Importance",
    scoring_metric_label: str = "Mean Decrease in PR-AUC (Average Precision)",
) -> Path:
    """Render and save a publication-quality horizontal bar chart of feature importance.

    Args:
        importance_df: DataFrame containing 'feature', 'importance_mean', and 'importance_std'.
        output_path: Target image filepath (PNG).
        title: Chart title.
        scoring_metric_label: Description of the metric drop plotted on x-axis.

    Returns:
        Path to saved figure.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Invert order for top-to-bottom bar plotting
    df_plot = importance_df.sort_values(by="importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    y_pos = np.arange(len(df_plot))
    colors = [
        "#0284c7" if m > 0.1 else "#38bdf8" if m > 0.01 else "#94a3b8"
        for m in df_plot["importance_mean"]
    ]

    bars = ax.barh(
        y_pos,
        df_plot["importance_mean"],
        xerr=df_plot["importance_std"],
        capsize=4,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.7,
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["feature"], fontsize=10, fontweight="medium")
    ax.set_xlabel(scoring_metric_label, fontsize=10, fontweight="medium", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Gridlines and aesthetics
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")

    # Extend x-axis limit so whiskers and annotations never overlap or get clipped
    max_reach = max(df_plot["importance_mean"] + df_plot["importance_std"])
    ax.set_xlim(0, max(max_reach * 1.18, 0.75))

    # Add value annotations outside the error bar whiskers
    for bar, mean, std in zip(bars, df_plot["importance_mean"], df_plot["importance_std"]):
        ax.annotate(
            f"{mean:.4f}",
            xy=(max(mean + std, 0) + 0.012, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
            color="#1e293b",
        )

    # Add scientific causation disclaimer as footnote
    fig.text(
        0.5,
        0.01,
        CAUSATION_DISCLAIMER,
        wrap=True,
        horizontalalignment="center",
        fontsize=7.5,
        color="#64748b",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def save_feature_importance_reports(
    importance_df: pd.DataFrame,
    tables_dir: Union[str, Path] = TABLES_DIR,
    figures_dir: Union[str, Path] = FIGURES_DIR,
) -> Dict[str, Path]:
    """Persist feature importance table as CSV and Markdown, and figure as PNG.

    Args:
        importance_df: Ranked feature importance DataFrame.
        tables_dir: Directory for tables.
        figures_dir: Directory for figures.

    Returns:
        Dictionary of written file paths.
    """
    t_dir = Path(tables_dir)
    f_dir = Path(figures_dir)
    t_dir.mkdir(parents=True, exist_ok=True)
    f_dir.mkdir(parents=True, exist_ok=True)

    csv_path = t_dir / "feature_importance.csv"
    md_path = t_dir / "feature_importance.md"
    png_path = f_dir / "feature_importance.png"

    # 1. Save CSV
    importance_df.to_csv(csv_path, index=False)

    # 2. Save Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MaintainIQ — Permutation Feature Importance Table\n\n")
        f.write(
            "> [!NOTE]\n"
            f"> **Scientific Note**: {CAUSATION_DISCLAIMER} "
            "Evaluated on held-out evaluation sample using 10 permutation repeats "
            "against the Average Precision (PR-AUC) metric.\n\n"
        )
        f.write(dataframe_to_markdown(importance_df))
        f.write("\n\n*Generated by MaintainIQ explainability module.*\n")

    # 3. Save publication-quality figure
    plot_permutation_importance(importance_df, png_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "figure": png_path,
    }


def run_explainability_pipeline(
    model_path: Path = MODEL_PATH,
    data_path: Path = RAW_DATA_FILE,
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute complete explainability workflow.

    Workflow:
    1. Loads saved model from model_path.
    2. Loads dataset from data_path, cleans and stratifies 80/20.
    3. Computes permutation importance on evaluation split (X_test, y_test).
    4. Computes SHAP attributions for sample records (normal and failure cases).
    5. Saves reports to reports/tables/ and reports/figures/.

    Returns:
        Summary dictionary with importance DataFrame and saved paths.
    """
    print(f"[MaintainIQ] Loading saved model from: {model_path}")
    pipeline = load_model(model_path)

    print(f"[MaintainIQ] Loading evaluation data from: {data_path}")
    raw_df = load_raw_data(data_path, validate=True)
    cleaned_df = clean_raw_data(raw_df)
    X, y = split_features_target(cleaned_df)

    _, X_test, _, y_test = create_stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("[MaintainIQ] Computing permutation feature importance across 10 repeats...")
    importance_df = compute_permutation_importance(
        pipeline=pipeline,
        X_eval=X_test,
        y_eval=y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=random_state,
        use_transformed_features=True,
    )

    print("[MaintainIQ] Saving importance tables and publication-quality figure...")
    saved_paths = save_feature_importance_reports(
        importance_df=importance_df,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    # Compute a local instance explanation on first test sample
    sample_record = X_test.iloc[[0]]
    local_explanation = explain_prediction(pipeline, sample_record)
    print(
        f"[MaintainIQ] Sample instance local explanation computed via {local_explanation['method']}."
    )

    return {
        "importance_df": importance_df,
        "sample_explanation": local_explanation,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    run_explainability_pipeline()
