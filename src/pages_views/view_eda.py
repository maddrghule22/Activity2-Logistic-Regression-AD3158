"""
Page 3: Exploratory Data Analysis (EDA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.ui_components import render_section_header
from src.visualizer import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_bivariate_scatter
)

def render_eda_page(df: pd.DataFrame):
    render_section_header("Exploratory Data Analysis (EDA)", "Discover feature distributions, academic patterns, correlations, and placement drivers")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['id', 'Student_ID']]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in ['Student_ID']]

    # Tabs for EDA aspects
    tab_dist, tab_corr, tab_bivariate, tab_academic, tab_outliers = st.tabs([
        "📈 Feature Distributions",
        "🔥 Correlation Heatmap",
        "🎯 Bivariate Relationships",
        "🎓 Academic & Skill Impact",
        "📦 Outlier Analysis"
    ])

    with tab_dist:
        st.markdown("#### Univariate Feature Distributions by Placement Status")
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_feature = st.selectbox("Select Feature to Visualize Distribution:", numeric_cols, index=0)
        with col_sel2:
            group_hue = st.selectbox("Group By / Hue:", ['Placement_Status'] + cat_cols, index=0)

        fig_dist = plot_feature_distribution(df, selected_feature, hue=group_hue)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Key Statistical Insights")
        if 'Placement_Status' in df.columns and selected_feature in df.columns:
            placed_vals = df[df['Placement_Status'] == 'Placed'][selected_feature].dropna()
            unplaced_vals = df[df['Placement_Status'] == 'Not Placed'][selected_feature].dropna()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"Mean {selected_feature} (Placed)", f"{placed_vals.mean():.2f}")
            with c2:
                st.metric(f"Mean {selected_feature} (Not Placed)", f"{unplaced_vals.mean():.2f}")
            with c3:
                diff = placed_vals.mean() - unplaced_vals.mean()
                st.metric("Academic Delta", f"{diff:+.2f}", delta=f"{diff:+.2f}")

    with tab_corr:
        st.markdown("#### Pearson Correlation Analysis")
        st.markdown("Calculates linear correlation coefficients $r \\in [-1, +1]$ across all numerical features to detect multicollinearity and strong placement predictors.")
        fig_corr = plot_correlation_heatmap(df, numeric_cols)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_bivariate:
        st.markdown("#### Bivariate Feature Scatter Interaction")
        col_x, col_y, col_h = st.columns(3)
        with col_x:
            feat_x = st.selectbox("X-Axis Feature:", numeric_cols, index=0 if len(numeric_cols)>0 else 0)
        with col_y:
            feat_y = st.selectbox("Y-Axis Feature:", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
        with col_h:
            hue_sel = st.selectbox("Color Hue:", ['Placement_Status'] + cat_cols, index=0)

        fig_scatter = plot_bivariate_scatter(df, feat_x, feat_y, hue_sel)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_academic:
        st.markdown("#### Academic & Industry Exposure Analysis")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("##### 1. CGPA vs Placement Outcome")
            if 'CGPA' in df.columns and 'Placement_Status' in df.columns:
                fig_cgpa = px.box(
                    df, x='Placement_Status', y='CGPA', color='Placement_Status',
                    points='all',
                    color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
                    title="CGPA Spread by Placement Status"
                )
                fig_cgpa.update_layout(template='plotly_white')
                st.plotly_chart(fig_cgpa, use_container_width=True)

        with col_a2:
            st.markdown("##### 2. Internships vs Placement Outcome")
            if 'Internships' in df.columns and 'Placement_Status' in df.columns:
                intern_ct = pd.crosstab(df['Internships'], df['Placement_Status'], normalize='index') * 100
                intern_ct = intern_ct.reset_index()
                fig_intern = px.bar(
                    intern_ct, x='Internships', y=['Not Placed', 'Placed'],
                    barmode='stack',
                    color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
                    title="Placement Probability by Number of Internships (%)"
                )
                fig_intern.update_layout(template='plotly_white', yaxis_title="Percentage (%)")
                st.plotly_chart(fig_intern, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("##### 3. Work Experience Impact")
            if 'Work_Experience' in df.columns and 'Placement_Status' in df.columns:
                work_ct = pd.crosstab(df['Work_Experience'], df['Placement_Status'], normalize='index') * 100
                fig_work = px.bar(
                    work_ct.reset_index(), x='Work_Experience', y=['Not Placed', 'Placed'],
                    barmode='group',
                    color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
                    title="Placement Rate by Prior Work Experience (%)"
                )
                fig_work.update_layout(template='plotly_white', yaxis_title="Percentage (%)")
                st.plotly_chart(fig_work, use_container_width=True)

        with col_b2:
            st.markdown("##### 4. Backlogs vs Placement")
            if 'Backlogs' in df.columns and 'Placement_Status' in df.columns:
                backlog_ct = pd.crosstab(df['Backlogs'], df['Placement_Status'], normalize='index') * 100
                fig_back = px.bar(
                    backlog_ct.reset_index(), x='Backlogs', y=['Not Placed', 'Placed'],
                    barmode='group',
                    color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
                    title="Placement Rate by Backlog Count (%)"
                )
                fig_back.update_layout(template='plotly_white', yaxis_title="Percentage (%)")
                st.plotly_chart(fig_back, use_container_width=True)

    with tab_outliers:
        st.markdown("#### Outlier Detection via Interquartile Range (IQR)")
        st.markdown("Identifies values falling beyond $Q_1 - 1.5\\times\\text{IQR}$ or $Q_3 + 1.5\\times\\text{IQR}$.")
        
        outlier_feat = st.selectbox("Select Feature for Outlier Boxplot:", numeric_cols, index=0)
        fig_box = px.box(
            df, y=outlier_feat, points="outliers",
            title=f"Boxplot & Outliers for {outlier_feat}",
            color_discrete_sequence=['#2563EB']
        )
        fig_box.update_layout(template='plotly_white')
        st.plotly_chart(fig_box, use_container_width=True)

        # Compute IQR statistics
        q1 = df[outlier_feat].quantile(0.25)
        q3 = df[outlier_feat].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((df[outlier_feat] < lower_bound) | (df[outlier_feat] > upper_bound)).sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1 (25th Percentile)", f"{q1:.2f}")
        c2.metric("Q3 (75th Percentile)", f"{q3:.2f}")
        c3.metric("IQR Range", f"{iqr:.2f}")
        c4.metric("Detected Outliers", f"{outlier_count}", delta=f"{outlier_count/len(df)*100:.1f}%")
