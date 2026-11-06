"""
Page 8: Prediction History & Database Audit
"""

import streamlit as st
import pandas as pd
from src.ui_components import render_section_header, render_metric_card
from src.db_manager import DatabaseManager

def render_history_page(db: DatabaseManager):
    render_section_header("Prediction History & Database Audit", "Track, search, filter, and audit all live student predictions stored in SQLite")

    records_df = db.get_all_records()
    stats = db.get_stats()

    # Metrics Summary Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Predictions Logged", f"{stats['total_predictions']:,}", "SQLite Stored Sessions")
    with c2:
        render_metric_card("Placed Inferences", f"{stats['placed_count']:,}", f"{stats['placed_rate']}% of queries")
    with c3:
        render_metric_card("At-Risk / Unlikely", f"{stats['not_placed_count']:,}", "Recommended for counseling")
    with c4:
        render_metric_card("Average Queried CGPA", f"{stats['avg_cgpa']:.2f}", "Across all entries")

    st.write("")

    if records_df.empty:
        st.info("ℹ️ No predictions have been logged in the database yet. Go to the **Student Prediction** tab to make live inferences!")
        return

    # Filter and Search Controls
    col_search, col_filter, col_dl, col_clear = st.columns([2, 1, 1, 1])
    
    with col_search:
        search_query = st.text_input("🔍 Search by Student ID / Roll No:", placeholder="e.g. STU_2026_01")
    with col_filter:
        filter_status = st.selectbox("Filter Outcome:", ["All", "Placed", "Not Placed"])
    with col_dl:
        csv_bytes = records_df.to_csv(index=False).encode('utf-8')
        st.write("")
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name="student_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_clear:
        st.write("")
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            db.clear_all()
            st.success("History successfully cleared!")
            st.rerun()

    # Apply Filters
    filtered_df = records_df.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df['student_identifier'].str.contains(search_query, case=False, na=False)]
    if filter_status != "All":
        filtered_df = filtered_df[filtered_df['prediction'] == filter_status]

    st.write("")
    st.markdown(f"**Showing {len(filtered_df)} of {len(records_df)} records:**")
    st.dataframe(filtered_df, use_container_width=True)
