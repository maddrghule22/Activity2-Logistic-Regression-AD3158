"""
Smart Student Performance & Placement Prediction System
Main Streamlit Application Entrypoint
"""

import os
import streamlit as st
import pandas as pd

from src.ui_components import inject_custom_css
from src.data_generator import generate_student_dataset
from src.preprocessor import DataPreprocessor
from src.model_engine import ModelEngine
from src.evaluator import ModelEvaluator
from src.db_manager import DatabaseManager

# Page Views
from src.pages_views.view_overview import render_overview_page
from src.pages_views.view_dataset import render_dataset_page
from src.pages_views.view_eda import render_eda_page
from src.pages_views.view_training import render_training_page
from src.pages_views.view_evaluation import render_evaluation_page
from src.pages_views.view_prediction import render_prediction_page
from src.pages_views.view_comparison import render_comparison_page
from src.pages_views.view_history import render_history_page
from src.pages_views.view_documentation import render_documentation_page

# Set Page Config
st.set_page_config(
    page_title="Student Placement Prediction System | Logistic Regression",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern styling
inject_custom_css()

# Session State Initialization
@st.cache_resource
def get_db_instance():
    return DatabaseManager()

def initialize_system():
    if 'db' not in st.session_state:
        st.session_state['db'] = get_db_instance()

    dataset_path = "data/student_placement_dataset.csv"
    if 'df' not in st.session_state:
        if os.path.exists(dataset_path):
            st.session_state['df'] = pd.read_csv(dataset_path)
        else:
            df = generate_student_dataset(n_samples=1200, random_state=42)
            os.makedirs("data", exist_ok=True)
            df.to_csv(dataset_path, index=False)
            st.session_state['df'] = df

    if 'pipeline' not in st.session_state or 'engine' not in st.session_state:
        df = st.session_state['df']
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_train_test_data(
            df, test_size=0.20, random_state=42
        )
        transformer = preprocessor.build_transformer()
        engine = ModelEngine(preprocessor_pipeline=transformer)
        
        model_file = "models/placement_logistic_model.joblib"
        if os.path.exists(model_file):
            try:
                saved_data = engine.load_model(model_file)
                pipeline = saved_data['pipeline']
                engine.trained_pipeline = pipeline
                engine.feature_names = saved_data.get('feature_names', [])
                engine.best_params = saved_data.get('best_params', {})
            except Exception:
                pipeline = engine.train_logistic_regression(X_train, y_train)
                engine.save_model(model_file)
        else:
            pipeline = engine.train_logistic_regression(X_train, y_train)
            engine.save_model(model_file)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        eval_res = ModelEvaluator.evaluate_predictions(y_test, y_proba, threshold=0.50)

        st.session_state['pipeline'] = pipeline
        st.session_state['engine'] = engine
        st.session_state['eval_results'] = eval_res
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

initialize_system()

# State update callbacks
def set_dataset(new_df: pd.DataFrame):
    st.session_state['df'] = new_df

def set_model_state(pipeline, eval_res, engine, X_train, X_test, y_train, y_test):
    st.session_state['pipeline'] = pipeline
    st.session_state['eval_results'] = eval_res
    st.session_state['engine'] = engine
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 16px 0;">
        <span style="font-size: 2.4rem;">🎓</span>
        <h2 style="margin: 4px 0 0 0; font-size: 1.25rem; font-weight: 800; color: #1e293b;">Placement ML Hub</h2>
        <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Logistic Regression Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page_options = [
        "1. Overview Dashboard",
        "2. Dataset Explorer",
        "3. Exploratory Data Analysis",
        "4. Model Training",
        "5. Model Evaluation",
        "6. Student Prediction",
        "7. Model Comparison",
        "8. Prediction History",
        "9. Project Documentation"
    ]
    
    selected_page = st.radio(
        "Navigation",
        page_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Sidebar Status Card
    if 'eval_results' in st.session_state:
        acc = st.session_state['eval_results'].get('accuracy', 0.89) * 100
        auc = st.session_state['eval_results'].get('auc_score', 0.96)
        db_stats = st.session_state['db'].get_stats()
        
        st.markdown(f"""
        <div style="background: #f1f5f9; border-radius: 10px; padding: 14px; font-size: 0.82rem; border: 1px solid #cbd5e1;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 6px;">⚡ Model Health Status</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">Model Type:</span>
                <span style="font-weight: 600; color: #1e293b;">Logit (Scikit-Learn)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">Test Accuracy:</span>
                <span style="font-weight: 600; color: #10b981;">{acc:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">ROC-AUC:</span>
                <span style="font-weight: 600; color: #2563eb;">{auc:.3f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">Logged Predictions:</span>
                <span style="font-weight: 600; color: #6366f1;">{db_stats['total_predictions']} queries</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 20px; font-size: 0.72rem; color: #94a3b8;">
        B.Tech AI & Data Science Project<br>
        Python 3.12 • Scikit-Learn • Streamlit
    </div>
    """, unsafe_allow_html=True)

# Page Routing
df = st.session_state['df']
pipeline = st.session_state.get('pipeline')
engine = st.session_state.get('engine')
eval_res = st.session_state.get('eval_results', {})
X_train = st.session_state.get('X_train')
X_test = st.session_state.get('X_test')
y_train = st.session_state.get('y_train')
y_test = st.session_state.get('y_test')
db = st.session_state['db']

if selected_page.startswith("1."):
    render_overview_page(df, pipeline, eval_res)
elif selected_page.startswith("2."):
    render_dataset_page(df, set_dataset)
elif selected_page.startswith("3."):
    render_eda_page(df)
elif selected_page.startswith("4."):
    render_training_page(df, pipeline, set_model_state)
elif selected_page.startswith("5."):
    render_evaluation_page(pipeline, X_test, y_test)
elif selected_page.startswith("6."):
    render_prediction_page(pipeline, db)
elif selected_page.startswith("7."):
    render_comparison_page(engine, X_train, X_test, y_train, y_test, df)
elif selected_page.startswith("8."):
    render_history_page(db)
elif selected_page.startswith("9."):
    render_documentation_page()
