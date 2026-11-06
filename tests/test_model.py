"""
Unit Tests for Model Engine and Evaluator
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import generate_student_dataset
from src.preprocessor import DataPreprocessor
from src.model_engine import ModelEngine
from src.evaluator import ModelEvaluator

@pytest.fixture
def trained_setup():
    df = generate_student_dataset(n_samples=200, random_state=42, add_noise_and_nulls=False)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_train_test_data(df, test_size=0.25)
    transformer = preprocessor.build_transformer()
    engine = ModelEngine(preprocessor_pipeline=transformer)
    pipeline = engine.train_logistic_regression(X_train, y_train)
    return engine, pipeline, X_test, y_test, X_train, y_train

def test_logistic_prediction(trained_setup):
    engine, pipeline, X_test, y_test, _, _ = trained_setup
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    
    assert len(probabilities) == len(X_test)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

def test_model_evaluation_metrics(trained_setup):
    _, pipeline, X_test, y_test, _, _ = trained_setup
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = ModelEvaluator.evaluate_predictions(y_test, probabilities, threshold=0.5)
    
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1_score'] <= 1.0
    assert 0.0 <= metrics['auc_score'] <= 1.0
    assert metrics['confusion_matrix']['tn'] + metrics['confusion_matrix']['fp'] + \
           metrics['confusion_matrix']['fn'] + metrics['confusion_matrix']['tp'] == len(y_test)

def test_model_comparison(trained_setup):
    engine, _, X_test, y_test, X_train, y_train = trained_setup
    comp_df, fitted_dict = engine.compare_models(X_train, X_test, y_train, y_test)
    
    assert 'Logistic Regression' in comp_df['Model'].values
    assert 'Decision Tree' in comp_df['Model'].values
    assert 'Random Forest' in comp_df['Model'].values
    assert len(comp_df) == 3
