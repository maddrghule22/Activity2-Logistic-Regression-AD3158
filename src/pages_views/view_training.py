"""
Page 4: Model Training
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import render_section_header, render_metric_card
from src.preprocessor import DataPreprocessor
from src.model_engine import ModelEngine
from src.evaluator import ModelEvaluator

def render_training_page(df: pd.DataFrame, current_pipeline, set_model_callback):
    render_section_header("Logistic Regression Model Training", "Configure hyperparameters, execute leakage-free pipeline training, and perform cross-validation")

    # Hyperparameter & Preprocessing Configuration
    st.markdown("### ⚙️ Pipeline Configuration")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        test_size = st.slider("Test Set Split Ratio:", min_value=0.10, max_value=0.40, value=0.20, step=0.05,
                              help="Fraction of dataset reserved for testing and evaluation (stratified).")
        scaling_choice = st.selectbox("Feature Scaling Method:", ["StandardScaler", "MinMaxScaler"],
                                     help="StandardScaler normalizes mean=0, std=1; MinMaxScaler bounds values [0, 1].")
    with col_c2:
        reg_penalty = st.selectbox("Regularization Penalty:", ["L2 (Ridge)", "L1 (Lasso)"], index=0,
                                   help="L2 penalizes large weights; L1 promotes sparsity/feature selection.")
        c_param = st.select_slider(
            "Inverse Regularization Strength (C):",
            options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            value=1.0,
            help="Smaller values specify stronger regularization."
        )
    with col_c3:
        if reg_penalty.startswith("L1"):
            solver_choice = st.selectbox("Optimization Solver:", ["liblinear", "saga"])
            penalty_val = "l1"
        else:
            solver_choice = st.selectbox("Optimization Solver:", ["lbfgs", "liblinear", "saga"])
            penalty_val = "l2"
        max_iter = st.number_input("Max Iterations:", min_value=100, max_value=5000, value=1000, step=100)

    # Grid Search tuning toggle
    tune_mode = st.checkbox("🔍 Enable Automated Hyperparameter Tuning (GridSearchCV)", value=False,
                            help="Searches optimal combinations of C and penalty using 5-Fold Stratified Cross-Validation.")

    st.write("")
    train_clicked = st.button("🚀 Train Model Pipeline", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner("Executing Data Preprocessing, Stratified Split, and Logistic Regression Training..."):
            try:
                # 1. Preprocess & Split
                preprocessor = DataPreprocessor(scaling_method=scaling_choice)
                X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_train_test_data(
                    df, test_size=test_size, random_state=42
                )
                transformer = preprocessor.build_transformer()
                
                # 2. Engine Initialization
                engine = ModelEngine(preprocessor_pipeline=transformer)
                
                if tune_mode:
                    tuning_res = engine.perform_hyperparameter_tuning(X_train, y_train, cv_folds=5)
                    pipeline = tuning_res['best_model']
                    st.success(f"GridSearchCV completed! Best F1 Score: {tuning_res['best_f1_score']:.4f}")
                    st.json(tuning_res['best_params'])
                else:
                    pipeline = engine.train_logistic_regression(
                        X_train, y_train,
                        C=c_param,
                        penalty=penalty_val,
                        solver=solver_choice,
                        max_iter=int(max_iter),
                        random_state=42
                    )
                
                # 3. Cross-Validation on training data
                cv_scores = engine.evaluate_cross_validation(X_train, y_train, cv_folds=5)
                
                # 4. Evaluate on test set
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                eval_res = ModelEvaluator.evaluate_predictions(y_test, y_proba, threshold=0.50)
                
                # Save to disk
                saved_path = engine.save_model("models/placement_logistic_model.joblib")
                
                # Update Streamlit global session state
                set_model_callback(pipeline, eval_res, engine, X_train, X_test, y_train, y_test)
                
                st.balloons()
                st.success(f"🎉 Model trained successfully and serialized to `{saved_path}`!")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    # Display Trained Pipeline Status and CV metrics
    if current_pipeline is not None and 'eval_results' in st.session_state:
        eval_res = st.session_state['eval_results']
        st.write("")
        st.markdown("### 🏆 Active Model Quick Metrics (Test Set)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{eval_res['accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{eval_res['precision'] * 100:.2f}%")
        m3.metric("Recall", f"{eval_res['recall'] * 100:.2f}%")
        m4.metric("F1 Score", f"{eval_res['f1_score'] * 100:.2f}%")
        m5.metric("ROC-AUC", f"{eval_res['auc_score']:.4f}")

        # Model download button
        if 'engine' in st.session_state:
            with open("models/placement_logistic_model.joblib", "rb") as f:
                model_bytes = f.read()
            st.download_button(
                label="💾 Download Trained Model (.joblib)",
                data=model_bytes,
                file_name="placement_logistic_model.joblib",
                mime="application/octet-stream"
            )

    # Theoretical Callout: Sigmoid and Mathematical Foundations
    st.write("")
    with st.expander("📖 Mathematical Theory: The Sigmoid Function & Log-Odds Formulation", expanded=True):
        st.markdown(r"""
        #### 1. The Log-Odds (Logit) Equation
        Logistic Regression models the probability $p = P(Y=1|X)$ by expressing the log-odds as an affine linear combination of input features:
        $$\ln\left(\frac{p}{1 - p}\right) = z = \beta_0 + \sum_{i=1}^n \beta_i x_i$$
        
        #### 2. The Sigmoid (Logistic) Transformation
        Solving for $p$ yields the **Sigmoid function** $\sigma(z)$, which maps any real-valued number $z \in (-\infty, +\infty)$ into a valid probability $p \in (0, 1)$:
        $$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + \exp\left(-\left(\beta_0 + \sum_{i=1}^n \beta_i x_i\right)\right)}$$

        #### 3. Binary Cross-Entropy (Log-Loss) Objective Function
        Parameters $\beta$ are estimated using Maximum Likelihood Estimation (MLE), minimizing:
        $$\mathcal{L}(\beta) = -\frac{1}{N}\sum_{j=1}^N \left[ y_j \ln(p_j) + (1 - y_j) \ln(1 - p_j) \right] + \lambda \cdot \mathcal{R}(\beta)$$
        where $\mathcal{R}(\beta) = \|\beta\|_2^2$ for L2 (Ridge) and $\|\beta\|_1$ for L1 (Lasso), with $\lambda = \frac{1}{C}$.
        """)
