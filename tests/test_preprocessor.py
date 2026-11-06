"""
Unit Tests for Data Preprocessor Module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_student_dataset
from src.preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    return generate_student_dataset(n_samples=100, random_state=42, add_noise_and_nulls=True)

def test_summary_generation(sample_data):
    summary = DataPreprocessor.get_dataset_summary(sample_data)
    assert 'rows' in summary
    assert 'columns' in summary
    assert summary['rows'] > 0
    assert summary['columns'] == 12
    assert summary['duplicates'] >= 0

def test_data_cleaning_and_split(sample_data):
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_train_test_data(
        sample_data, test_size=0.25, random_state=42
    )
    assert len(X_train) + len(X_test) <= len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert len(features) > 5

def test_pipeline_transformation(sample_data):
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_train_test_data(sample_data)
    transformer = preprocessor.build_transformer()
    
    transformed_train = transformer.fit_transform(X_train)
    transformed_test = transformer.transform(X_test)
    
    assert transformed_train.shape[0] == len(X_train)
    assert transformed_test.shape[0] == len(X_test)
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert not np.isnan(transformed_train).any()
