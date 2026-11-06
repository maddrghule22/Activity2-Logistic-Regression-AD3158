"""
Page 7: Model Comparison & Advanced Decision Boundary Visualizer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
from src.ui_components import render_section_header
from src.evaluator import ModelEvaluator
from src.visualizer import (
    plot_model_comparison_bars,
    plot_coefficients_importance,
    plot_decision_boundary_surface
)

def render_comparison_page(engine, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, df: pd.DataFrame):
    render_section_header("Model Comparison & Decision Boundaries", "Benchmark Logistic Regression against non-linear Decision Tree and Random Forest classifiers")

    if engine is None or X_train is None or X_test is None:
        st.warning("⚠️ Training data or model engine not initialized. Please train a model in the Model Training page first.")
        return

    tab_comp, tab_coef, tab_boundary = st.tabs([
        "📊 Multi-Model Benchmarking",
        "⚖️ Coefficients & Odds Ratios",
        "🗺️ 2D Decision Boundary Contour"
    ])

    with tab_comp:
        st.markdown("#### Performance Metrics Comparison (Test Set)")
        with st.spinner("Training baseline Decision Tree and Random Forest on identical train-test splits..."):
            comp_df, fitted_pipelines = engine.compare_models(X_train, X_test, y_train, y_test)

        st.dataframe(comp_df, use_container_width=True)

        # Bar chart comparison
        fig_bar = plot_model_comparison_bars(comp_df)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Overlaid ROC Curves
        st.markdown("#### Comparative ROC Curves")
        fig_roc_comp = go.Figure()
        
        colors = {'Logistic Regression': '#2563EB', 'Decision Tree': '#F59E0B', 'Random Forest': '#10B981'}
        for name, pipe in fitted_pipelines.items():
            proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = roc_auc_score(y_test, proba)
            fig_roc_comp.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} (AUC = {auc_val:.3f})",
                line=dict(color=colors.get(name, '#64748B'), width=2.5)
            ))
            
        fig_roc_comp.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='#94A3B8')))
        fig_roc_comp.update_layout(
            template='plotly_white',
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=30, b=40, l=40, r=40),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_roc_comp, use_container_width=True)

    with tab_coef:
        st.markdown("#### Logistic Regression Feature Coefficients & Odds Ratios")
        st.markdown(r"""
        The **Odds Ratio (OR)** represents the multiplicative change in the odds of placement for a 1-unit increase in the standardized feature:
        $$\text{Odds Ratio} = \exp(\beta_i)$$
        - $\text{OR} > 1.0$: Increases the odds of placement (Positive Driver).
        - $\text{OR} < 1.0$: Decreases the odds of placement (Negative Driver / Risk Factor).
        """)
        try:
            df_coef, intercept = engine.get_feature_coefficients()
            st.metric("Model Bias Intercept (\u03b2\u2080)", f"{intercept:.4f}")
            
            c1, c2 = st.columns([3, 2])
            with c1:
                fig_coef = plot_coefficients_importance(df_coef)
                st.plotly_chart(fig_coef, use_container_width=True)
            with c2:
                st.dataframe(df_coef, use_container_width=True)
        except Exception as e:
            st.error(f"Could not extract coefficients: {e}")

    with tab_boundary:
        st.markdown("#### Interactive 2D Decision Boundary Surface")
        st.markdown("Select any two continuous academic attributes. The system computes a 2D grid of probabilities while holding other features constant at their median/mode.")

        numeric_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
        c_b1, c_b2, c_b3 = st.columns(3)
        with c_b1:
            feat_x = st.selectbox("Feature 1 (X-Axis):", numeric_cols, index=0)
        with c_b2:
            feat_y = st.selectbox("Feature 2 (Y-Axis):", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        with c_b3:
            b_thresh = st.slider("Boundary Threshold Isocline:", 0.10, 0.90, 0.50, 0.05)

        with st.spinner("Generating 2D meshgrid surface across pipeline transformation..."):
            xx, yy, zz = ModelEvaluator.generate_decision_boundary_2d(
                engine.trained_pipeline,
                X_train,
                feat_x=feat_x,
                feat_y=feat_y,
                grid_resolution=50
            )
            fig_boundary = plot_decision_boundary_surface(
                xx, yy, zz, feat_x, feat_y, df, threshold=b_thresh
            )
            st.plotly_chart(fig_boundary, use_container_width=True)
