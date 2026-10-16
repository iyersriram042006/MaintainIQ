"""MaintainIQ — ML-Based Predictive Maintenance System.

Interactive Streamlit dashboard providing:
1. Overview & Dataset Context (UCI AI4I 2020 Predictive Maintenance).
2. Single Machine Risk Prediction with Real-Time Physical Indicators & SHAP Attributions.
3. Batch CSV Telemetry Evaluation with Validation & Downloadable Reports.
4. Model Benchmarking & Metric Comparison (Logistic Regression, Decision Tree, KNN, SVM).
5. Explainability & Permutation Feature Importance.
6. Optional Syllabus Experimental Extensions (K-Means Clustering & Linear Regression).
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    ALL_EXPECTED_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    RAW_DATA_FILE,
    TABLES_DIR,
    TARGET_COLUMN,
)
from src.data_loader import load_raw_data
from src.explain import CAUSATION_DISCLAIMER, explain_prediction
from src.predict import (
    ensure_model_pipeline,
    evaluate_batch_predictions,
    extract_actual_failure_column,
    load_model,
    predict_batch_dataframe,
    predict_failure_risk,
    validate_upload_schema,
)

# -----------------------------------------------------------------------------
# Canonical Benchmark and Feature Importance Defaults (Cloud Deployment Fallback)
# -----------------------------------------------------------------------------
CANONICAL_BENCHMARK_ROWS = [
    {
        "Model": "decision_tree",
        "PR-AUC": 0.7291454013443732,
        "F1-Score": 0.5643564356435643,
        "Precision": 0.4253731343283582,
        "Recall": 0.8382352941176471,
        "Accuracy": 0.9560,
        "ROC-AUC": 0.9096600596760445,
        "TP": 57,
        "FP": 77,
        "FN": 11,
        "TN": 1855,
        "Latency (ms/sample)": 0.0009437085100216791,
        "Training Time (s)": 0.018147416005376726,
        "Imbalance Strategy": "class_weight='balanced' (cost-sensitive tree split reweighting)",
    },
    {
        "Model": "svm",
        "PR-AUC": 0.6119367651810571,
        "F1-Score": 0.4250871080139373,
        "Precision": 0.2785388127853881,
        "Recall": 0.8970588235294118,
        "Accuracy": 0.9175,
        "ROC-AUC": 0.9654655340397028,
        "TP": 61,
        "FP": 158,
        "FN": 7,
        "TN": 1774,
        "Latency (ms/sample)": 0.03986733350029681,
        "Training Time (s)": 1.0987406659987755,
        "Imbalance Strategy": "class_weight='balanced', probability=True (margin-weighted RBF support vectors)",
    },
    {
        "Model": "knn",
        "PR-AUC": 0.5515918238990715,
        "F1-Score": 0.43478260869565216,
        "Precision": 0.8333333333333334,
        "Recall": 0.29411764705882354,
        "Accuracy": 0.9740,
        "ROC-AUC": 0.8565757824869079,
        "TP": 20,
        "FP": 4,
        "FN": 48,
        "TN": 1928,
        "Latency (ms/sample)": 0.006463728990638629,
        "Training Time (s)": 0.008338250016095117,
        "Imbalance Strategy": "distance-weighted neighbors (weights='distance', k=7)",
    },
    {
        "Model": "logistic_regression",
        "PR-AUC": 0.38176196610560775,
        "F1-Score": 0.24190064794816415,
        "Precision": 0.14177215189873418,
        "Recall": 0.8235294117647058,
        "Accuracy": 0.8245,
        "ROC-AUC": 0.9069540859822189,
        "TP": 56,
        "FP": 339,
        "FN": 12,
        "TN": 1593,
        "Latency (ms/sample)": 0.001086541495169513,
        "Training Time (s)": 0.03610800000024028,
        "Imbalance Strategy": "class_weight='balanced' (inverse class frequency reweighting)",
    },
]

CANONICAL_FEATURE_IMPORTANCE_ROWS = [
    {"feature": "Torque [Nm]", "importance_mean": 0.5353822940957836, "importance_std": 0.019845359582705623},
    {"feature": "Air temperature [K]", "importance_mean": 0.500547059117967, "importance_std": 0.037281311189640884},
    {"feature": "Rotational speed [rpm]", "importance_mean": 0.4876523191717295, "importance_std": 0.014903595680900095},
    {"feature": "Tool wear [min]", "importance_mean": 0.26866464066901746, "importance_std": 0.021379341221766514},
    {"feature": "Process temperature [K]", "importance_mean": 0.25609364793818784, "importance_std": 0.018356099279716768},
    {"feature": "Type_L", "importance_mean": 0.055914303931631895, "importance_std": 0.03327134788238639},
    {"feature": "Type_M", "importance_mean": 0.0, "importance_std": 0.0},
    {"feature": "Type_H", "importance_mean": 0.0, "importance_std": 0.0},
]

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MaintainIQ — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Cached Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_model():
    """Load and cache the trained model pipeline.

    If models/model.joblib is missing (e.g. on Streamlit Community Cloud),
    automatically provisions the champion Decision Tree pipeline using the official dataset.
    """
    try:
        if not MODEL_PATH.exists():
            with st.spinner("⚙️ Setting up MaintainIQ champion model pipeline (Decision Tree)..."):
                return ensure_model_pipeline(MODEL_PATH)
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading or provisioning model: {e}")
        return None


@st.cache_data
def get_cached_dataset_stats():
    """Load basic metadata and statistics from the raw AI4I dataset."""
    if not RAW_DATA_FILE.exists():
        return None
    try:
        df = load_raw_data(RAW_DATA_FILE, validate=False)
        total = len(df)
        failures = int(df[TARGET_COLUMN].sum())
        normal = total - failures
        failure_rate = (failures / total) * 100.0
        return {
            "total_records": total,
            "normal_count": normal,
            "failure_count": failures,
            "failure_rate": failure_rate,
            "columns": list(df.columns),
            "sample_head": df.head(5),
        }
    except Exception:
        return None


@st.cache_data
def get_cached_comparison_reports():
    """Load generated benchmark metrics and comparison tables."""
    csv_path = TABLES_DIR / "model_comparison.csv"
    json_path = TABLES_DIR / "model_metrics.json"

    df_comp = None
    metrics_dict = None

    if csv_path.exists():
        try:
            df_comp = pd.read_csv(csv_path)
        except Exception:
            df_comp = pd.DataFrame(CANONICAL_BENCHMARK_ROWS)
    else:
        df_comp = pd.DataFrame(CANONICAL_BENCHMARK_ROWS)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
        except Exception:
            metrics_dict = None

    return df_comp, metrics_dict


@st.cache_data
def get_cached_feature_importance():
    """Load ranked permutation feature importance table."""
    csv_path = TABLES_DIR / "feature_importance.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame(CANONICAL_FEATURE_IMPORTANCE_ROWS)
    return pd.DataFrame(CANONICAL_FEATURE_IMPORTANCE_ROWS)



# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.title("⚙️ MaintainIQ")
    st.caption(
        "Applied Machine Learning for Industrial Telemetry & Predictive Maintenance | UCI AI4I 2020"
    )

    # Load artifacts
    model = get_cached_model()
    dataset_stats = get_cached_dataset_stats()
    comparison_df, metrics_dict = get_cached_comparison_reports()
    importance_df = get_cached_feature_importance()

    # System Status Warning if model missing
    if model is None:
        st.warning(
            "⚠️ **Model artifact not found** at `models/model.joblib`. "
            "Please run `python -m src.train` in your terminal to train and save the pipeline."
        )

    # Main Tabs Navigation
    tabs = st.tabs(
        [
            "📊 System Overview",
            "⚙️ Single Machine Risk Scoring",
            "📁 CSV Batch Evaluation",
            "📈 Model Benchmarks",
            "🔍 Explainability & SHAP",
            "🧪 Syllabus Experimental Extensions",
        ]
    )

    # =========================================================================
    # TAB 1: SYSTEM OVERVIEW
    # =========================================================================
    with tabs[0]:
        st.subheader("Industrial Predictive Maintenance System")

        st.markdown(
            """
            **MaintainIQ** is an end-to-end machine learning system designed to forecast 
            milling machine component failure risk from real-time operational telemetry.
            By continuously monitoring mechanical stress, cutting tool degradation, and thermal 
            gradients, maintenance teams can schedule interventions **before** catastrophic breakdown occurs.
            """
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_str = f"{dataset_stats['total_records']:,}" if dataset_stats else "10,000"
            st.metric("Dataset Records", total_str, "UCI AI4I 2020")
        with col2:
            fail_str = f"{dataset_stats['failure_rate']:.2f}%" if dataset_stats else "3.39%"
            st.metric("Failure Incidence", fail_str, "Severe Class Imbalance")
        with col3:
            best_model_name = "DECISION TREE"
            best_prauc = "0.7291"
            if comparison_df is not None and not comparison_df.empty:
                best_model_name = str(comparison_df.iloc[0]["Model"]).upper()
                best_prauc = f"{float(comparison_df.iloc[0]['PR-AUC']):.4f}"
            st.metric("Best Model", best_model_name, f"PR-AUC: {best_prauc}")
        with col4:
            st.metric("Evaluation Strategy", "PR-AUC & F1", "Stratified 80/20")

        st.markdown("---")

        col_left, col_right = st.columns([1.1, 0.9])
        with col_left:
            st.markdown("### 🔬 Industrial Failure Modes Monitored")
            st.write(
                """
                The system monitors continuous physical signals to detect degradation associated with 
                five distinct equipment failure modes:
                - **TWF (Tool Wear Failure)**: Cutter friction wear reaching physical limits (typically > 200–240 minutes).
                - **HDF (Heat Dissipation Failure)**: Thermal dissipation breakdown when the difference between process and air temperature falls below critical levels at low operating speeds.
                - **PWF (Power Failure)**: Overload or stalling when motor power (Torque × Rotational Speed) exceeds safe operating limits (< 3500 W or > 9000 W).
                - **OSF (Overstrain Failure)**: Combined product of excessive cutting torque and accumulated tool wear exceeding material fatigue thresholds.
                - **RNF (Random Failure)**: Uncorrelated stochastic hardware failures.
                """
            )

        with col_right:
            st.markdown("### ⚠️ Dataset Disclosure & Ethics")
            st.info(
                """
                **Synthetic Dataset Disclosure**:
                This project utilizes the **UCI AI4I 2020 Predictive Maintenance Dataset**, 
                which is generated from a phenomenological physical simulator reflecting 
                milling machine dynamics.
                
                **Educational Purpose**:
                MaintainIQ is built as a college engineering mini-project demonstrating applied 
                machine learning and pipeline safety. Predictions reflect empirical statistical 
                reliance rather than absolute real-world mechanical guarantees.
                """
            )

        if dataset_stats and "sample_head" in dataset_stats:
            with st.expander("👁️ Preview Raw UCI Dataset Telemetry (First 5 Records)"):
                st.dataframe(dataset_stats["sample_head"], use_container_width=True)

    # =========================================================================
    # TAB 2: SINGLE MACHINE PREDICTION
    # =========================================================================
    with tabs[1]:
        st.subheader("Single Machine Failure Risk Scoring")
        st.write(
            "Input operational telemetry for the **six approved feature variables** to evaluate failure risk."
        )

        # Quick preset buttons for demo
        st.markdown("**Quick Preset Scenarios:**")
        preset_cols = st.columns(4)
        preset_choice = None
        if preset_cols[0].button("🟢 Normal Machine", use_container_width=True):
            preset_choice = "normal"
        if preset_cols[1].button("🔴 High Overstrain (OSF)", use_container_width=True):
            preset_choice = "overstrain"
        if preset_cols[2].button("🔴 High Power (PWF)", use_container_width=True):
            preset_choice = "power"
        if preset_cols[3].button("🔴 Heat Dissipation (HDF)", use_container_width=True):
            preset_choice = "heat"

        # Preset values dictionary
        presets = {
            "normal": {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1550,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 15,
            },
            "overstrain": {
                "Type": "L",
                "Air temperature [K]": 302.5,
                "Process temperature [K]": 311.5,
                "Rotational speed [rpm]": 1350,
                "Torque [Nm]": 68.0,
                "Tool wear [min]": 225,
            },
            "power": {
                "Type": "L",
                "Air temperature [K]": 299.0,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 2860,
                "Torque [Nm]": 75.0,
                "Tool wear [min]": 60,
            },
            "heat": {
                "Type": "H",
                "Air temperature [K]": 304.5,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1250,
                "Torque [Nm]": 58.0,
                "Tool wear [min]": 140,
            },
        }

        active_vals = presets.get(preset_choice, presets["normal"])

        col_input, col_result = st.columns([1, 1.1])

        with col_input:
            st.markdown("#### Operational Parameters")
            with st.form("single_prediction_form"):
                type_idx = ["L", "M", "H"].index(active_vals["Type"])
                input_type = st.selectbox(
                    "Product Quality Variant (Type)",
                    options=["L", "M", "H"],
                    index=type_idx,
                    help="L: Low (50%), M: Medium (30%), H: High variant precision (20%)",
                )

                input_air_temp = st.slider(
                    "Air Temperature [K]",
                    min_value=290.0,
                    max_value=310.0,
                    value=float(active_vals["Air temperature [K]"]),
                    step=0.1,
                    help="Ambient factory room temperature in Kelvin (dataset range: ~295–305 K)",
                )

                input_process_temp = st.slider(
                    "Process Temperature [K]",
                    min_value=300.0,
                    max_value=320.0,
                    value=float(active_vals["Process temperature [K]"]),
                    step=0.1,
                    help="Tool-workpiece interface temperature in Kelvin (dataset range: ~305–315 K)",
                )

                input_speed = st.slider(
                    "Rotational Speed [rpm]",
                    min_value=1000,
                    max_value=3000,
                    value=int(active_vals["Rotational speed [rpm]"]),
                    step=10,
                    help="Spindle rotational velocity in rpm (dataset range: 1168–2886 rpm)",
                )

                input_torque = st.slider(
                    "Torque [Nm]",
                    min_value=3.0,
                    max_value=85.0,
                    value=float(active_vals["Torque [Nm]"]),
                    step=0.5,
                    help="Motor spindle resistance torque in Newton-meters (dataset range: 3.8–76.6 Nm)",
                )

                input_tool_wear = st.slider(
                    "Tool Wear [min]",
                    min_value=0,
                    max_value=260,
                    value=int(active_vals["Tool wear [min]"]),
                    step=1,
                    help="Cumulative cutting tool usage in minutes (dataset range: 0–253 min)",
                )

                submitted = st.form_submit_button("⚡ Evaluate Machine Health", use_container_width=True)

        with col_result:
            st.markdown("#### Health Assessment & Risk Scoring")

            sample_dict = {
                "Type": input_type,
                "Air temperature [K]": input_air_temp,
                "Process temperature [K]": input_process_temp,
                "Rotational speed [rpm]": input_speed,
                "Torque [Nm]": input_torque,
                "Tool wear [min]": input_tool_wear,
            }

            if model is not None:
                prediction_res = predict_failure_risk(model, sample_dict)
                pred_label = prediction_res["machine_failure_prediction"]
                prob = prediction_res["failure_probability"]
                risk_pct = prediction_res["risk_percentage"]
                risk_band = prediction_res["risk_level"]

                # Prominent Visual Status Card
                if risk_band == "High":
                    st.error(
                        f"🚨 **CRITICAL RISK ALERT** — Machine Failure Probable ({risk_pct:.1f}%)"
                    )
                elif risk_band == "Medium":
                    st.warning(
                        f"⚠️ **ELEVATED RISK WARNING** — Schedule Inspection ({risk_pct:.1f}%)"
                    )
                else:
                    st.success(
                        f"✅ **NORMAL OPERATIONAL STATE** — Low Risk ({risk_pct:.1f}%)"
                    )

                # Metrics Row
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Predicted State", "FAILURE (1)" if pred_label == 1 else "NORMAL (0)")
                m_col2.metric("Failure Probability", f"{prob:.4f}")
                m_col3.metric("Risk Classification", risk_band)

                st.progress(min(prob, 1.0), text=f"Failure Probability: {risk_pct:.1f}%")

                # Physical Domain Heuristics
                st.markdown("##### ⚙️ Physical Telemetry Indicators")
                temp_delta = input_process_temp - input_air_temp
                mech_power = input_torque * (input_speed * (2 * np.pi / 60))
                overstrain = input_torque * input_tool_wear

                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric(
                    "Temp Delta (ΔT)",
                    f"{temp_delta:.1f} K",
                    "Low ΔT = HDF risk" if temp_delta < 8.6 else "Adequate",
                )
                p_col2.metric(
                    "Spindle Power",
                    f"{mech_power:.0f} W",
                    "Out of bounds = PWF" if (mech_power < 3500 or mech_power > 9000) else "Optimal",
                )
                p_col3.metric(
                    "Overstrain (τ × Wear)",
                    f"{overstrain:,.0f}",
                    "Fatigue limit warning" if overstrain > 11000 else "Nominal",
                )

                # Local SHAP Contribution
                with st.expander("🔍 Local Feature Contributions (SHAP / Tree Attribution)"):
                    try:
                        explanation = explain_prediction(model, sample_dict)
                        attrs = explanation["feature_attributions"]
                        st.caption(f"Calculated via **{explanation['method']}**:")

                        # Plot local attributions
                        fig_local, ax_local = plt.subplots(figsize=(6, 3), dpi=150)
                        features_local = list(reversed(list(attrs.keys())[:6]))
                        vals_local = [attrs[f] for f in features_local]
                        bar_colors = ["#ef4444" if v > 0 else "#10b981" for v in vals_local]

                        ax_local.barh(features_local, vals_local, color=bar_colors, alpha=0.85)
                        ax_local.axvline(0, color="#64748b", linestyle="--", linewidth=0.8)
                        ax_local.set_xlabel("Impact on Failure Log-Odds (SHAP Value)", fontsize=8)
                        ax_local.tick_params(labelsize=8)
                        ax_local.spines["top"].set_visible(False)
                        ax_local.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig_local)
                        plt.close(fig_local)
                        st.caption("🔴 Red bars increase failure risk; 🟢 green bars reduce risk.")
                    except Exception as ex:
                        st.info(f"Local SHAP attribution could not be rendered: {ex}")

                st.caption(f"ℹ️ {CAUSATION_DISCLAIMER}")
            else:
                st.info("Train the model to enable live predictions.")

    # =========================================================================
    # TAB 3: CSV BATCH EVALUATION
    # =========================================================================
    with tabs[2]:
        st.subheader("Batch Equipment Telemetry Evaluation")
        st.write(
            "Upload a CSV file containing machine sensor logs to evaluate an entire equipment fleet."
        )

        # Template download helper
        template_df = pd.DataFrame(
            [
                {
                    "Type": "M",
                    "Air temperature [K]": 298.1,
                    "Process temperature [K]": 308.6,
                    "Rotational speed [rpm]": 1551,
                    "Torque [Nm]": 42.8,
                    "Tool wear [min]": 0,
                },
                {
                    "Type": "L",
                    "Air temperature [K]": 302.5,
                    "Process temperature [K]": 311.5,
                    "Rotational speed [rpm]": 1350,
                    "Torque [Nm]": 68.0,
                    "Tool wear [min]": 225,
                },
            ]
        )
        template_csv = template_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=template_csv,
            file_name="maintainiq_template.csv",
            mime="text/csv",
            help="Download a pre-formatted template with the six required feature columns.",
        )

        uploaded_file = st.file_uploader(
            "Choose a telemetry CSV file",
            type=["csv"],
            help="Upload a CSV with the 6 approved features. Target and ID columns are NOT required.",
        )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

                if user_df.empty:
                    st.error("Uploaded CSV file is empty. Please provide a valid file.")
                elif not validate_upload_schema(user_df):
                    missing = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
                    st.error(
                        f"❌ **Schema Validation Failed!** Uploaded CSV is missing required columns: `{missing}`.\n\n"
                        f"Expected exactly these 6 feature columns: `{FEATURE_COLUMNS}`."
                    )
                elif model is None:
                    st.warning("Model pipeline is not loaded. Cannot process batch predictions.")
                else:
                    st.success(
                        f"✅ **Schema Verified!** Successfully loaded {len(user_df):,} records for evaluation."
                    )

                    with st.spinner("Executing batch inference across pipeline..."):
                        annotated_df = predict_batch_dataframe(model, user_df)

                    actual_col = extract_actual_failure_column(user_df)

                    if actual_col is not None:
                        batch_metrics = evaluate_batch_predictions(annotated_df, actual_col)

                        st.info(
                            f"🎯 **Ground-Truth Failure Labels Detected (`{actual_col}`)**\n\n"
                            "Displaying separate actual versus predicted breakdown. "
                            "**Predictions are model inferences and are strictly evaluated against ground truth, never treated as actual outcomes.**"
                        )

                        st.markdown("##### Ground Truth vs. Model Predictions")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Batch Records Evaluated", f"{batch_metrics['total_samples']:,}")
                        c2.metric(
                            "Actual Failures (Ground Truth)",
                            f"{batch_metrics['actual_failures']:,}",
                            f"{batch_metrics['actual_failure_rate']:.2f}% true failure rate",
                        )
                        c3.metric(
                            "Predicted Failures (Model Output)",
                            f"{batch_metrics['predicted_failures']:,}",
                            f"{batch_metrics['predicted_failure_rate']:.2f}% flagged by model",
                        )
                        c4.metric(
                            "High-Risk Alerts",
                            f"{int((annotated_df['risk_level'] == 'High').sum()):,}",
                            "Urgent Inspection Required",
                        )

                        st.markdown("##### Batch Classification Performance")
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric(
                            "Recall (Detection Rate)",
                            f"{batch_metrics['recall']:.4f}",
                            f"Captured {batch_metrics['tp']}/{batch_metrics['actual_failures']} failures",
                        )
                        p2.metric(
                            "Precision (Alert Reliability)",
                            f"{batch_metrics['precision']:.4f}",
                            f"{batch_metrics['fp']} False Positives",
                        )
                        p3.metric("F1-Score", f"{batch_metrics['f1']:.4f}")
                        p4.metric("Accuracy", f"{batch_metrics['accuracy']:.4f}")

                        with st.expander("📊 Uploaded Batch Confusion Matrix Detail"):
                            cm1, cm2, cm3, cm4 = st.columns(4)
                            cm1.metric("True Positives (TP)", str(batch_metrics["tp"]))
                            cm2.metric("False Negatives (FN)", str(batch_metrics["fn"]))
                            cm3.metric("False Positives (FP)", str(batch_metrics["fp"]))
                            cm4.metric("True Negatives (TN)", str(batch_metrics["tn"]))
                    else:
                        st.caption(
                            "ℹ️ **Unlabeled Operational Telemetry**: Uploaded file contains sensor features without ground-truth failure labels. "
                            "Metrics below represent model inferences and operational risk classifications, not observed physical outcomes."
                        )
                        total_batch = len(annotated_df)
                        pred_fails = int((annotated_df["predicted_failure"] == 1).sum())
                        high_risk = int((annotated_df["risk_level"] == "High").sum())
                        norm_count = total_batch - pred_fails

                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Machines Evaluated", f"{total_batch:,}")
                        b2.metric("Predicted Failures (Model Output)", f"{pred_fails:,}", f"{(pred_fails/total_batch)*100:.1f}%")
                        b3.metric("High-Risk Alerts", f"{high_risk:,}", "Immediate Action")
                        b4.metric("Predicted Normal", f"{norm_count:,}")

                    # Preview table
                    st.markdown("##### Annotated Fleet Predictions")
                    st.dataframe(annotated_df.head(25), use_container_width=True)

                    # Download annotated CSV
                    out_csv = annotated_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Annotated Predictions CSV",
                        data=out_csv,
                        file_name="maintainiq_batch_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as ex:
                st.error(f"Error reading or processing uploaded CSV: {ex}")

    # =========================================================================
    # TAB 4: MODEL BENCHMARKS
    # =========================================================================
    with tabs[3]:
        st.subheader("Model Benchmarks & Classifier Comparison")
        st.write(
            "Empirical results obtained from the **held-out 80/20 stratified test set** "
            "(2,000 samples, ~3.39% failures). Models fit exclusively on training data."
        )

        if comparison_df is not None and not comparison_df.empty:
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=["PR-AUC", "F1-Score", "Precision", "Recall", "ROC-AUC"],
                    color="#bbf7d0",
                ),
                use_container_width=True,
            )

            # Comparative Bar Chart
            st.markdown("#### Primary Metric Benchmarks across Classifiers")
            chart_metrics = ["PR-AUC", "F1-Score", "Recall", "Precision", "Accuracy", "ROC-AUC"]
            plot_df = comparison_df.set_index("Model")[[m for m in chart_metrics if m in comparison_df.columns]]
            st.bar_chart(plot_df)

            # Confusion Matrix Detail
            st.markdown("#### Test Set Confusion Matrix Breakdown")
            cm_cols = st.columns(len(comparison_df))
            for i, (_, row) in enumerate(comparison_df.iterrows()):
                with cm_cols[i]:
                    st.markdown(f"**{row['Model'].upper()}**")
                    st.write(f"- **True Positives (TP)**: {int(row['TP'])}")
                    st.write(f"- **False Negatives (FN)**: {int(row['FN'])}")
                    st.write(f"- **False Positives (FP)**: {int(row['FP'])}")
                    st.write(f"- **True Negatives (TN)**: {int(row['TN'])}")
                    st.caption(f"Latency: {row['Latency (ms/sample)']*1000:.2f} µs/sample")

            with st.expander("📘 Class Imbalance Mitigation Strategy Breakdown"):
                st.markdown(
                    """
                    1. **Decision Tree**: Configured with `class_weight='balanced'`, `max_depth=8`, `min_samples_leaf=4`. Achieved highest PR-AUC (0.7291) and optimal harmonic balance of failure detection (Recall 83.82%) and precision (42.54%). Selected as champion model.
                    2. **Support Vector Machine (SVM)**: RBF kernel with `class_weight='balanced'` and probability estimation. Achieved highest ROC-AUC (0.9655) and highest raw failure recall (89.71%, detecting 61 of 68 test failures).
                    3. **K-Nearest Neighbors (KNN)**: Distance-weighted neighborhood (`k=7`). Achieved highest precision (83.33%) with only 4 false alarms, but lower recall (29.41%).
                    4. **Logistic Regression**: Linear baseline with `class_weight='balanced'`. High recall (82.35%) with significant false positives (339 FP, precision 14.18%), demonstrating that non-linear decision boundaries are required for physical telemetry modeling.
                    """
                )
        else:
            st.info(
                "No comparison report found at `reports/tables/model_comparison.csv`. "
                "Run `python -m src.train` to generate benchmark tables."
            )

    # =========================================================================
    # TAB 5: EXPLAINABILITY & SHAP
    # =========================================================================
    with tabs[4]:
        st.subheader("Model Interpretability & Permutation Feature Importance")
        st.write(
            "Evaluation of feature reliance using permutation importance on the held-out test split."
        )

        st.warning(f"⚠️ **Scientific Disclaimer**: {CAUSATION_DISCLAIMER}")

        fig_path = FIGURES_DIR / "feature_importance.png"
        col_fig, col_tbl = st.columns([1.1, 0.9])

        with col_fig:
            if fig_path.exists():
                st.image(
                    str(fig_path),
                    caption="Permutation Feature Importance (PR-AUC Metric Drop)",
                    use_container_width=True,
                )
            elif importance_df is not None and not importance_df.empty:
                from src.explain import plot_permutation_importance
                try:
                    plot_permutation_importance(importance_df, fig_path)
                    st.image(
                        str(fig_path),
                        caption="Permutation Feature Importance (PR-AUC Metric Drop)",
                        use_container_width=True,
                    )
                except Exception:
                    st.info("Run `python -m src.explain` to generate the feature importance figure.")
            else:
                st.info("Run `python -m src.explain` to generate the feature importance figure.")

        with col_tbl:
            st.markdown("#### Ranked Feature Importance Table")
            if importance_df is not None and not importance_df.empty:
                display_imp = pd.DataFrame(
                    {
                        "Rank": list(range(1, len(importance_df) + 1)),
                        "Feature": importance_df["feature"],
                        "PR-AUC Drop": importance_df["importance_mean"].map(lambda x: f"{x:.4f}"),
                        "Std Dev": importance_df["importance_std"].map(lambda x: f"±{x:.4f}"),
                    }
                )
                st.table(display_imp, hide_index=True)
            else:
                st.info("Feature importance CSV not found. Run `python -m src.explain`.")

            st.markdown("#### Engineering Domain Insights")
            st.markdown(
                """
                - **Torque [Nm] & Rotational Speed [rpm]**: These parameters govern motor mechanical power ($P = \\tau \\cdot \\omega$) and are the dominant predictors of **Power Failure (PWF)** and **Overstrain Failure (OSF)**.
                - **Air & Process Temperature [K]**: The thermal gradient $\\Delta T$ directly determines **Heat Dissipation Failure (HDF)** risk.
                - **Tool Wear [min]**: Linear accumulation of cutter contact triggers **Tool Wear Failure (TWF)** and accelerates overstrain.
                - **Type Variants (L, M, H)**: Demonstrate minimal direct impact compared to real-time physical sensor signals.
                """
            )

    # =========================================================================
    # TAB 6: SYLLABUS EXPERIMENTAL EXTENSIONS
    # =========================================================================
    with tabs[5]:
        st.subheader("Optional Syllabus Experimental Extensions")
        st.markdown(
            """
            *Note on Academic Scope*: These experimental modules utilize algorithms permitted
            by the academic syllabus (**K-Means Clustering** and **Linear Regression**)
            in clearly separated tasks that do **not** replace the primary binary failure classification models:
            - **K-Means Clustering**: Unsupervised machine working regime grouping.
            - **Linear Regression**: Continuous physical parameter estimation (Cutting Torque).
            """
        )

        sub_tab1, sub_tab2 = st.tabs(["🧩 K-Means Regime Clustering", "📐 Linear Regression Modeling"])

        with sub_tab1:
            st.markdown("#### Unsupervised Equipment Operating Regime Segmentation (K-Means)")
            st.write(
                "Groups machines into distinct operational clusters based on continuous physical telemetry."
            )
            n_clusters = st.slider("Select Number of Operating Clusters (k)", min_value=2, max_value=6, value=3)

            if RAW_DATA_FILE.exists():
                try:
                    from src.experimental import cluster_operating_regimes
                    raw_df_exp = load_raw_data(RAW_DATA_FILE, validate=False)
                    num_data = raw_df_exp[NUMERICAL_FEATURES].dropna()
                    _, clustered_df, summary = cluster_operating_regimes(num_data, n_clusters=n_clusters)

                    st.success(f"Successfully clustered {len(clustered_df):,} records into {n_clusters} operating regimes (Inertia: {summary['inertia']:.1f}).")

                    profile_rows = []
                    for cid, prof in summary["profiles"].items():
                        profile_rows.append({
                            "Cluster": f"Regime {cid}",
                            "Machine Count": prof["sample_count"],
                            "Avg Torque [Nm]": f"{prof['mean_torque']:.1f}",
                            "Avg Speed [rpm]": f"{prof['mean_speed']:.0f}",
                            "Avg Tool Wear [min]": f"{prof['mean_tool_wear']:.1f}",
                            "Avg ΔT [K]": f"{prof['mean_temp_delta']:.2f}",
                        })
                    st.table(pd.DataFrame(profile_rows))
                except Exception as ex:
                    st.error(f"Error running K-Means clustering: {ex}")
            else:
                st.info("Raw dataset required for clustering.")

        with sub_tab2:
            st.markdown("#### Continuous Physical Torque Modeling (Linear Regression)")
            st.write(
                "Predicts cutting **Torque [Nm]** from spindle speed and thermal variables using Ordinary Least Squares."
            )
            if RAW_DATA_FILE.exists():
                try:
                    from src.experimental import train_torque_regression
                    raw_df_exp = load_raw_data(RAW_DATA_FILE, validate=False)
                    _, reg_metrics = train_torque_regression(raw_df_exp[NUMERICAL_FEATURES].dropna())

                    r1, r2, r3 = st.columns(3)
                    r1.metric("R² Score", f"{reg_metrics['r2_score']:.4f}", "Variance Explained")
                    r2.metric("Mean Absolute Error (MAE)", f"{reg_metrics['mae_nm']:.2f} Nm", "Torque Accuracy")
                    r3.metric("Root Mean Squared Error (RMSE)", f"{reg_metrics['rmse_nm']:.2f} Nm")

                    st.caption("Demonstrates syllabus-permitted Linear Regression for continuous physical estimation without target leakage into binary failure classification.")
                except Exception as ex:
                    st.error(f"Error training Linear Regression model: {ex}")
            else:
                st.info("Raw dataset required for regression.")


if __name__ == "__main__":
    main()
