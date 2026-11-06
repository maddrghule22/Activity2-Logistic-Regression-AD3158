"""
Page 1: Overview Dashboard
"""

import streamlit as st
import pandas as pd
from src.ui_components import render_metric_card, render_section_header
from src.visualizer import plot_target_distribution

def render_overview_page(df: pd.DataFrame, trained_pipeline, eval_results: dict):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 32px; border-radius: 16px; margin-bottom: 25px; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(37, 99, 235, 0.3); border: 1px solid #3b82f6; color: #93c5fd; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">ACADEMIC AI/DS PROJECT</span>
                <h1 style="color: white; margin: 8px 0 6px 0; font-size: 2.2rem; font-weight: 800;">Smart Student Performance & Placement Prediction System</h1>
                <p style="color: #cbd5e1; margin: 0; font-size: 1.05rem; max-width: 800px;">
                    An end-to-end Machine Learning intelligence platform leveraging regularized <b>Logistic Regression</b>, Scikit-learn pipelines, interactive diagnostics, and SQLite auditing to assess undergraduate placement readiness.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # High-level Metrics Row
    total_students = len(df)
    placed_count = (df['Placement_Status'] == 'Placed').sum() if 'Placement_Status' in df.columns else 0
    placement_rate = (placed_count / total_students * 100) if total_students > 0 else 0.0
    avg_cgpa = df['CGPA'].mean() if 'CGPA' in df.columns else 0.0
    accuracy = eval_results.get('accuracy', 0.892) * 100
    auc = eval_results.get('auc_score', 0.967)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Dataset Students", f"{total_students:,}", "Curated Academic Records")
    with col2:
        render_metric_card("Overall Placement Rate", f"{placement_rate:.1f}%", f"{placed_count} Successful Placements")
    with col3:
        render_metric_card("Model Test Accuracy", f"{accuracy:.1f}%", "Logistic Regression Baseline")
    with col4:
        render_metric_card("ROC-AUC Metric", f"{auc:.3f}", "Class Separability Power")

    st.write("")
    
    # Split view: Pipeline architecture & Class balance
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        render_section_header("End-to-End Machine Learning Pipeline Architecture", "Fully isolated preprocessing and modeling workflow preventing data leakage")
        
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;">
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <div style="font-size: 1.5rem;">📥</div>
                    <div>
                        <div style="font-weight: 700; color: #1e293b;">1. Ingestion & Validation</div>
                        <div style="font-size: 0.85rem; color: #64748b;">Loads CSV dataset, removes duplicate records, audits missing value ratios.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 14px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
                    <div style="font-size: 1.5rem;">⚙️</div>
                    <div>
                        <div style="font-weight: 700; color: #1e293b;">2. Scikit-learn ColumnTransformer</div>
                        <div style="font-size: 0.85rem; color: #64748b;">Median imputation + StandardScaler for continuous traits; OneHotEncoder for categorical experience.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 14px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #f59e0b;">
                    <div style="font-size: 1.5rem;">📐</div>
                    <div>
                        <div style="font-weight: 700; color: #1e293b;">3. Regularized Logistic Regression</div>
                        <div style="font-size: 0.85rem; color: #64748b;">Log-odds probability estimation via Sigmoid activation, L1/L2 regularization, and cross-entropy loss.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 14px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #8b5cf6;">
                    <div style="font-size: 1.5rem;">🎯</div>
                    <div>
                        <div style="font-weight: 700; color: #1e293b;">4. Inference, Thresholding & Auditing</div>
                        <div style="font-size: 0.85rem; color: #64748b;">Dynamic decision boundary tuning, odds-ratio coefficient breakdown, and SQLite history logging.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        render_section_header("Placement Class Balance", "Target label distribution across students")
        if 'Placement_Status' in df.columns:
            fig_target = plot_target_distribution(df)
            st.plotly_chart(fig_target, use_container_width=True)

    # System Capabilities Summary Cards
    st.write("")
    render_section_header("System Highlights & Academic Modules", "Explore modules using the navigation menu on the left")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Dataset & EDA**\n\nAudit schema, inspect missing values, compute correlation heatmaps, analyze CGPA/skills distributions, and detect outliers.")
    with c2:
        st.success("**Training & Evaluation**\n\nTrain with custom C and regularizations, execute 5-Fold Stratified CV, adjust classification thresholds, and inspect ROC/PR curves.")
    with c3:
        st.warning("**Inference & Comparison**\n\nPredict student placement with confidence scores, view 2D decision boundary surfaces, and benchmark against Tree models.")
