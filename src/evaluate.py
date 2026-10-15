"""Model evaluation utilities for MaintainIQ.

Computes class-imbalance-appropriate classification metrics and exports
comparison reports in both human-readable Markdown and machine-readable JSON.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import TABLES_DIR


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_pred: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    inference_time_sec: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute performance metrics for predictive maintenance binary classification.

    Focuses on imbalance-robust metrics (PR-AUC, Recall, Precision, F1, ROC-AUC)
    rather than misleading raw accuracy.

    Args:
        y_true: Ground truth binary failure labels (0: Normal, 1: Failure).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Positive-class failure probabilities (values between 0.0 and 1.0).
        inference_time_sec: Total inference execution time in seconds.
        n_samples: Number of samples evaluated (defaults to len(y_true)).

    Returns:
        Dictionary containing precision, recall, f1_score, pr_auc, roc_auc,
        confusion matrix breakdowns, and inference latency.

    Raises:
        ValueError: If y_true and y_pred have mismatched lengths or invalid values.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} entries while "
            f"y_pred has {len(y_pred_arr)} entries."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    # Validate binary values
    unique_true = set(np.unique(y_true_arr))
    if not unique_true.issubset({0, 1}):
        raise ValueError(f"y_true must contain only binary 0 and 1, got: {unique_true}")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    metrics: Dict[str, Any] = {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "sample_count": int(n_samples or len(y_true_arr)),
        "failure_count": int(np.sum(y_true_arr)),
    }

    # Probability-based metrics (PR-AUC / Average Precision and ROC-AUC)
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        if len(y_prob_arr) != len(y_true_arr):
            raise ValueError(
                f"Length mismatch: y_prob has {len(y_prob_arr)} entries while "
                f"y_true has {len(y_true_arr)} entries."
            )

        # ROC-AUC and PR-AUC require at least one positive and one negative sample
        if len(unique_true) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
            metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_prob_arr))
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    # Inference timing metrics
    if inference_time_sec is not None:
        count = n_samples or len(y_true_arr)
        total_ms = float(inference_time_sec * 1000.0)
        metrics["inference_time_total_ms"] = total_ms
        metrics["inference_latency_per_sample_ms"] = float(total_ms / max(count, 1))

    return metrics


def evaluate_model_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained scikit-learn pipeline on test features.

    Measures end-to-end inference latency, computes predictions and probabilities,
    and returns comprehensive metrics.

    Args:
        pipeline: Trained scikit-learn Pipeline.
        X_test: Test features DataFrame.
        y_test: Ground truth test labels.

    Returns:
        Evaluation metrics dictionary.
    """
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inf_time_sec = time.perf_counter() - t0

    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    return evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        inference_time_sec=inf_time_sec,
        n_samples=len(X_test),
    )


def build_comparison_dataframe(
    model_evaluations: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Format multiple model evaluation results into a clean comparison DataFrame.

    Args:
        model_evaluations: Dictionary mapping model names to their metrics dicts.

    Returns:
        DataFrame sorted by PR-AUC descending.
    """
    rows = []
    for model_name, metrics in model_evaluations.items():
        cm = metrics.get("confusion_matrix", {})
        row = {
            "Model": model_name,
            "PR-AUC": metrics.get("pr_auc"),
            "F1-Score": metrics.get("f1_score"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "ROC-AUC": metrics.get("roc_auc"),
            "TP": cm.get("true_positives"),
            "FP": cm.get("false_positives"),
            "FN": cm.get("false_negatives"),
            "TN": cm.get("true_negatives"),
            "Latency (ms/sample)": metrics.get("inference_latency_per_sample_ms"),
            "Imbalance Strategy": metrics.get("imbalance_strategy", "N/A"),
        }
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    if "PR-AUC" in df_comp.columns and not df_comp["PR-AUC"].isnull().all():
        df_comp = df_comp.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    return df_comp


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row:
            if pd.isna(val):
                formatted_vals.append("N/A")
            elif isinstance(val, (float, np.floating)):
                formatted_vals.append(f"{val:.4f}")
            else:
                formatted_vals.append(str(val))
        rows.append("| " + " | ".join(formatted_vals) + " |")

    return "\n".join([header_line, separator_line] + rows)


def save_evaluation_reports(
    comparison_df: pd.DataFrame,
    full_metrics: Dict[str, Any],
    output_dir: Path = TABLES_DIR,
) -> Dict[str, Path]:
    """Save comparison table (Markdown, CSV) and machine-readable metrics (JSON).

    Args:
        comparison_df: Tabular summary DataFrame.
        full_metrics: Complete detailed metrics dictionary.
        output_dir: Directory to save report files.

    Returns:
        Dictionary mapping report keys to written Path objects.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "model_comparison.csv"
    md_path = out_path / "model_comparison.md"
    json_path = out_path / "model_metrics.json"

    # 1. Save CSV
    comparison_df.to_csv(csv_path, index=False)

    # 2. Save Markdown Table with contextual header
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MaintainIQ — Model Evaluation & Comparison Table\n\n")
        f.write(
            "> [!NOTE]\n"
            "> **Dataset Note**: The UCI AI4I 2020 Predictive Maintenance Dataset "
            "is synthetic yet physically realistic. The test set comprises an 80/20 "
            "stratified split (2,000 samples, ~3.39% failure incidence). "
            "Because of the severe class imbalance, **PR-AUC and F1-Score** are prioritized "
            "over raw accuracy.\n\n"
        )
        f.write(dataframe_to_markdown(comparison_df))
        f.write("\n\n*Generated by MaintainIQ automated evaluation pipeline.*\n")

    # 3. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "json": json_path,
    }
