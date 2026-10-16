"""Tests for model evaluation metrics, reports, and comparison tables."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    build_comparison_dataframe,
    evaluate_predictions,
    save_evaluation_reports,
)


def test_evaluate_predictions_perfect_case():
    """Verify evaluation on a perfect prediction scenario."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.95]

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob, inference_time_sec=0.01)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 0
    assert "inference_time_total_ms" in metrics
    assert "inference_latency_per_sample_ms" in metrics


def test_evaluate_predictions_zero_division():
    """Verify that 0 positive predictions yield 0 precision without ZeroDivisionError."""
    y_true = [0, 1]
    y_pred = [0, 0]
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["roc_auc"] is None


def test_evaluate_predictions_length_mismatch():
    """Verify that length mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions([0, 1], [0])


def test_evaluate_predictions_empty_array():
    """Verify that empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="empty predictions array"):
        evaluate_predictions([], [])


def test_evaluate_predictions_invalid_labels():
    """Verify that non-binary target labels raise ValueError."""
    with pytest.raises(ValueError, match="must contain only binary"):
        evaluate_predictions([0, 2], [0, 1])


def test_build_comparison_dataframe():
    """Verify ranking and structure of model comparison table."""
    mock_evals = {
        "model_A": {
            "pr_auc": 0.50,
            "f1_score": 0.60,
            "precision": 0.70,
            "recall": 0.55,
            "roc_auc": 0.85,
            "confusion_matrix": {"true_positives": 10, "false_positives": 2, "false_negatives": 8, "true_negatives": 80},
            "inference_latency_per_sample_ms": 0.02,
            "imbalance_strategy": "strategy A",
        },
        "model_B": {
            "pr_auc": 0.80,  # Higher PR-AUC
            "f1_score": 0.75,
            "precision": 0.78,
            "recall": 0.72,
            "roc_auc": 0.92,
            "confusion_matrix": {"true_positives": 14, "false_positives": 4, "false_negatives": 4, "true_negatives": 78},
            "inference_latency_per_sample_ms": 0.05,
            "imbalance_strategy": "strategy B",
        },
    }

    df_comp = build_comparison_dataframe(mock_evals)
    assert len(df_comp) == 2
    # Model B should be ranked first because of higher PR-AUC
    assert df_comp.iloc[0]["Model"] == "model_B"
    assert df_comp.iloc[1]["Model"] == "model_A"
    assert "TP" in df_comp.columns
    assert "FP" in df_comp.columns
    assert "Accuracy" in df_comp.columns


def test_save_evaluation_reports(tmp_path: Path):
    """Verify export of Markdown, CSV, and JSON evaluation reports."""
    df_comp = pd.DataFrame([{"Model": "TestModel", "PR-AUC": 0.85, "F1-Score": 0.70}])
    metrics = {"TestModel": {"pr_auc": 0.85, "f1_score": 0.70}}

    report_paths = save_evaluation_reports(df_comp, metrics, output_dir=tmp_path)
    assert report_paths["csv"].exists()
    assert report_paths["markdown"].exists()
    assert report_paths["json"].exists()

    with open(report_paths["markdown"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "TestModel" in content
        assert "AI4I 2020" in content
