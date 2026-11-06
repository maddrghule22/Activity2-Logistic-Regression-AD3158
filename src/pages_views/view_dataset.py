"""
Page 2: Dataset Explorer
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import render_metric_card, render_section_header
from src.preprocessor import DataPreprocessor

def render_dataset_page(df: pd.DataFrame, set_dataset_callback):
    render_section_header("Dataset Explorer & Management", "Inspect raw data, handle missing values, remove duplicates, and upload custom datasets")

    # Data Source Selection
    st.markdown("### 📂 Dataset Source")
    src_choice = st.radio(
        "Choose Data Source:",
        ["Default Curated Placement Dataset (1,200 records)", "Upload Custom CSV Dataset"],
        horizontal=True
    )
    
    current_df = df
    
    if src_choice == "Upload Custom CSV Dataset":
        uploaded_file = st.file_uploader("Upload CSV File (must contain Placement_Status or target column):", type=["csv"])
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                st.success(f"Custom CSV uploaded successfully! Shape: {custom_df.shape[0]} rows, {custom_df.shape[1]} columns.")
                current_df = custom_df
                set_dataset_callback(custom_df)
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

    # Dataset Summary Metrics
    summary = DataPreprocessor.get_dataset_summary(current_df)
    memory_kb = current_df.memory_usage(deep=True).sum() / 1024
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Observations", f"{summary['rows']:,}", "Rows in current dataframe")
    with c2:
        render_metric_card("Total Features", f"{summary['columns']}", f"{len(summary['numeric_columns'])} Numeric, {len(summary['categorical_columns'])} Categorical")
    with c3:
        render_metric_card("Duplicate Rows", f"{summary['duplicates']}", "Redundant observations")
    with c4:
        render_metric_card("Memory Footprint", f"{memory_kb:.1f} KB", "In-memory RAM allocation")

    st.write("")
    
    # Cleaning & Deduplication actions
    col_clean1, col_clean2 = st.columns([2, 1])
    with col_clean1:
        if summary['duplicates'] > 0:
            st.warning(f"⚠️ Found {summary['duplicates']} duplicate records in the dataset.")
            if st.button("🧹 Remove Duplicate Records", use_container_width=True):
                cleaned = current_df.drop_duplicates().reset_index(drop=True)
                set_dataset_callback(cleaned)
                st.success(f"Removed duplicates! New dataset length: {len(cleaned)} rows.")
                st.rerun()
        else:
            st.success("✅ No duplicate records detected in the active dataset.")
            
    with col_clean2:
        csv_data = current_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset CSV",
            data=csv_data,
            file_name="student_placement_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Interactive Data Table Preview
    st.write("")
    st.markdown("### 🔍 Raw Dataset Sample (First 15 Rows)")
    st.dataframe(current_df.head(15), use_container_width=True)

    # Tabs for Schema, Missing Values, and Descriptive Statistics
    tab_stats, tab_missing, tab_schema = st.tabs(["📊 Descriptive Statistics", "⚠️ Missing Values Audit", "📋 Column Schema & Dtypes"])
    
    with tab_stats:
        st.markdown("#### Numerical Features Statistical Summary")
        numeric_df = current_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().transpose()
            # Add skewness
            stats['skewness'] = numeric_df.skew().round(3)
            stats = stats.round(2)
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("No numerical features found.")
            
        st.markdown("#### Categorical Features Frequency")
        cat_df = current_df.select_dtypes(exclude=[np.number])
        if not cat_df.empty:
            cat_stats = cat_df.describe().transpose()
            st.dataframe(cat_stats, use_container_width=True)

    with tab_missing:
        st.markdown("#### Missing Value Analysis")
        missing_df = summary['missing_summary']
        st.dataframe(missing_df, use_container_width=True)
        total_missing = missing_df['Missing_Count'].sum()
        if total_missing > 0:
            st.info(f"💡 The machine learning preprocessor pipeline automatically applies **Median Imputation** for numerical features and **Mode Imputation** for categorical features to avoid information loss.")
        else:
            st.success("✅ Dataset has 0 missing values.")

    with tab_schema:
        st.markdown("#### Column Data Types & Specifications")
        dtypes_df = pd.DataFrame({
            'Column Name': list(summary['dtypes'].keys()),
            'Data Type': list(summary['dtypes'].values()),
            'Feature Category': [
                'Numerical (Continuous)' if col in summary['numeric_columns'] and 'Score' in col or 'CGPA' in col
                else 'Numerical (Discrete)' if col in summary['numeric_columns']
                else 'Categorical (Binary)' if col in ['Work_Experience', 'Extracurricular_Activities', 'Placement_Status']
                else 'Identifier / Categorical'
                for col in summary['dtypes'].keys()
            ]
        })
        st.dataframe(dtypes_df, use_container_width=True)
