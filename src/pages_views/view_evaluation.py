"""
Page 5: Model Evaluation Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import render_section_header, render_metric_card
from src.evaluator import ModelEvaluator
from src.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_threshold_sensitivity
)

def render_evaluation_page(pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    render_section_header("Model Evaluation Dashboard", "Comprehensive diagnostics, threshold optimization, confusion matrix, and ROC/PR analysis")

    if pipeline is None or X_test is None or y_test is None:
        st.warning("⚠️ No trained model or test data found in active session. Please train the model in the Model Training page first.")
        return

    # Compute Test Set Probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Dynamic Threshold Slider
    st.markdown("### 🎚️ Interactive Classification Decision Threshold")
    st.markdown("Adjust the classification threshold $\\theta$ to observe real-time trade-offs between **Precision** (avoiding false placement promises) and **Recall** (identifying every placeable student).")

    col_th1, col_th2 = st.columns([3, 1])
    with col_th1:
        current_threshold = st.slider(
            "Classification Decision Threshold (default = 0.50):",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01
        )
    with col_th2:
        st.metric("Active Threshold", f"{current_threshold:.2f}")

    # Evaluate at the selected threshold
    eval_res = ModelEvaluator.evaluate_predictions(y_test, y_proba, threshold=current_threshold)

    # Classification Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Accuracy", f"{eval_res['accuracy'] * 100:.2f}%", "Overall Correct")
    with c2:
        render_metric_card("Precision", f"{eval_res['precision'] * 100:.2f}%", "TP / (TP + FP)")
    with c3:
        render_metric_card("Recall (Sensitivity)", f"{eval_res['recall'] * 100:.2f}%", "TP / (TP + FN)")
    with c4:
        render_metric_card("F1 Score", f"{eval_res['f1_score'] * 100:.2f}%", "Harmonic Mean")
    with c5:
        render_metric_card("Specificity", f"{eval_res['specificity'] * 100:.2f}%", "TN / (TN + FP)")

    st.write("")

    # Visualizations Row: Confusion Matrix & ROC Curve
    col_cm, col_roc = st.columns(2)
    with col_cm:
        fig_cm = plot_confusion_matrix(eval_res['confusion_matrix'])
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_roc:
        fig_roc = plot_roc_curve(eval_res['roc_curve'], eval_res['auc_score'])
        st.plotly_chart(fig_roc, use_container_width=True)

    # Row 2: Precision-Recall Curve & Threshold Sensitivity Curve
    col_pr, col_sens = st.columns(2)
    with col_pr:
        fig_pr = plot_pr_curve(eval_res['pr_curve'])
        st.plotly_chart(fig_pr, use_container_width=True)
    with col_sens:
        thresh_df = ModelEvaluator.compute_threshold_curve(np.array(y_test), y_proba, steps=60)
        fig_sens = plot_threshold_sensitivity(thresh_df, current_th=current_threshold)
        st.plotly_chart(fig_sens, use_container_width=True)

    # Classification Report & Academic Explanations
    st.write("")
    st.markdown("### 📋 Detailed Classification Report")
    st.dataframe(eval_res['classification_report_df'], use_container_width=True)

    with st.expander("🎓 Academic Metric Guide: Clinical & Placement Significance", expanded=False):
        st.markdown(r"""
        | Metric | Mathematical Formula | Placement Context Meaning |
        | :--- | :--- | :--- |
        | **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | Overall percentage of students correctly categorized. |
        | **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Of all students predicted as *Placed*, how many genuinely received offers? High precision prevents overpromising. |
        | **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Of all students who were actually capable of placement, what fraction did the model identify? High recall ensures no promising student is overlooked. |
        | **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing precision and recall under class imbalance. |
        | **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Ability to correctly identify students who require remedial intervention before campus recruitment. |
        | **ROC-AUC** | Area under FPR vs TPR curve | Scale-invariant measure of ranking ability across all decision boundaries. |
        """)
