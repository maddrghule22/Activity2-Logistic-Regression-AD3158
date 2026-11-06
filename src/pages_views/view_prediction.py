"""
Page 6: Interactive Student Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import render_section_header, render_prediction_badge
from src.db_manager import DatabaseManager
from src.visualizer import plot_prediction_gauge

def render_prediction_page(pipeline, db: DatabaseManager, default_threshold: float = 0.50):
    render_section_header("Interactive Student Placement Predictor", "Enter student academic and skill metrics to compute placement likelihood, confidence, and action recommendations")

    if pipeline is None:
        st.warning("⚠️ No active model pipeline found. Please train a model in the Model Training tab first.")
        return

    # Student Input Form
    with st.form("student_prediction_form"):
        st.markdown("#### 🎓 Student Profile Input")
        
        col_id, col_th = st.columns([2, 1])
        with col_id:
            student_id = st.text_input("Student Identifier / Roll No:", value="STU_2026_01")
        with col_th:
            decision_threshold = st.slider("Classification Threshold:", 0.10, 0.90, default_threshold, 0.05)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            cgpa = st.slider("Cumulative GPA (CGPA):", 5.0, 10.0, 8.2, 0.05,
                             help="Academic GPA on a 10.0 scale.")
            prog_score = st.slider("Programming Proficiency Score (0 - 100):", 0, 100, 78, 1,
                                   help="Coding assessment score (DSA, Problem Solving).")
            comm_score = st.slider("Communication Skills Score (0 - 100):", 0, 100, 75, 1,
                                   help="Verbal presentation & interview soft skills.")

        with c2:
            apt_score = st.slider("Aptitude / Quantitative Score (0 - 100):", 0, 100, 80, 1,
                                  help="Quantitative and logical reasoning score.")
            internships = st.number_input("Number of Internships:", min_value=0, max_value=6, value=2, step=1)
            projects = st.number_input("Major & Minor Projects:", min_value=0, max_value=10, value=3, step=1)

        with c3:
            certs = st.number_input("Technical Certifications:", min_value=0, max_value=8, value=2, step=1)
            work_exp = st.selectbox("Previous Work Experience?", ["Yes", "No"], index=0)
            backlogs = st.number_input("Academic Backlogs:", min_value=0, max_value=5, value=0, step=1)
            extracurricular = st.selectbox("Extracurricular Activities?", ["Yes", "No"], index=0)

        predict_button = st.form_submit_button("🔮 Predict Placement Likelihood", type="primary", use_container_width=True)

    if predict_button:
        # Prepare single record input
        input_dict = {
            'CGPA': float(cgpa),
            'Programming_Score': float(prog_score),
            'Communication_Score': float(comm_score),
            'Aptitude_Score': float(apt_score),
            'Internships': int(internships),
            'Projects': int(projects),
            'Technical_Certifications': int(certs),
            'Work_Experience': str(work_exp),
            'Backlogs': int(backlogs),
            'Extracurricular_Activities': str(extracurricular)
        }
        input_df = pd.DataFrame([input_dict])

        # Inference
        prob = float(pipeline.predict_proba(input_df)[0, 1])
        prediction = 'Placed' if prob >= decision_threshold else 'Not Placed'
        confidence = prob * 100 if prediction == 'Placed' else (1.0 - prob) * 100

        # Save to SQLite
        try:
            db.log_prediction(
                student_identifier=student_id,
                features=input_dict,
                prediction=prediction,
                probability=prob,
                confidence=confidence,
                threshold=decision_threshold
            )
            st.toast(f"Prediction logged to SQLite database for {student_id}", icon="✅")
        except Exception as e:
            st.error(f"Failed to log into database: {e}")

        # Display Result
        st.write("")
        render_prediction_badge(prediction, prob, confidence)

        col_g, col_rec = st.columns([1, 1])
        with col_g:
            fig_gauge = plot_prediction_gauge(prob, threshold=decision_threshold)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_rec:
            st.markdown("#### 💡 AI Academic Advisory & Remedial Insights")
            recommendations = []
            if backlogs > 0:
                recommendations.append(f"⚠️ **Clear Active Backlogs:** You currently have {backlogs} backlog(s). Clearing them provides an immediate positive log-odds boost.")
            if cgpa < 7.5:
                recommendations.append("📚 **Academic Elevation:** Increasing CGPA above 7.5 substantially enhances screening probability in tier-1 recruitment drives.")
            if prog_score < 70:
                recommendations.append("💻 **DSA & Coding Polish:** Target competitive programming platforms to elevate programming score above 75.")
            if internships == 0:
                recommendations.append("🏢 **Practical Industry Exposure:** Complete at least 1 summer or virtual technical internship to boost candidacy.")
            if projects < 2:
                recommendations.append("🛠️ **Portfolio Projects:** Build at least 2 full-stack or data-driven end-to-end projects with public GitHub repos.")
            
            if not recommendations:
                recommendations.append("🌟 **Outstanding Profile:** Strong academic record, rich skill scores, and solid project exposure. High likelihood of placement across core technology recruiters!")

            for rec in recommendations:
                st.markdown(f"- {rec}")

        # Feature Contribution Breakdown
        st.write("")
        st.markdown("#### 🔍 Feature Contribution Breakdown")
        
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'coef_'):
            # Transform input features to get normalized values
            preprocessor = pipeline.named_steps['preprocessor']
            transformed_x = preprocessor.transform(input_df)[0]
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
            
            coefs = classifier.coef_[0]
            contributions = transformed_x * coefs
            
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'Transformed_Value': np.round(transformed_x, 3),
                'Coefficient': np.round(coefs, 3),
                'Log_Odds_Contribution': np.round(contributions, 3),
                'Impact': ['Positive Impact (+)' if c >= 0 else 'Negative Impact (-)' for c in contributions]
            }).sort_values(by='Log_Odds_Contribution', ascending=False)
            
            st.dataframe(contrib_df, use_container_width=True)

        # Download Report Button
        st.write("")
        report_text = f"""==================================================
STUDENT PLACEMENT PREDICTION REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================
Student ID:            {student_id}
Prediction Outcome:    {prediction.upper()}
Placement Probability: {prob * 100:.2f}%
Model Confidence:      {confidence:.2f}%
Threshold Applied:     {decision_threshold:.2f}

PROFILE SUMMARY:
- CGPA:                 {cgpa}
- Programming Score:    {prog_score}
- Communication Score:  {comm_score}
- Aptitude Score:       {apt_score}
- Internships:          {internships}
- Projects:             {projects}
- Certifications:       {certs}
- Work Experience:      {work_exp}
- Backlogs:             {backlogs}
- Extracurriculars:     {extracurricular}
==================================================
"""
        st.download_button(
            label="📄 Download Student Prediction Report (.txt)",
            data=report_text,
            file_name=f"{student_id}_placement_report.txt",
            mime="text/plain"
        )
