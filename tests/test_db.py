"""
Unit Tests for Database Manager
"""

import os
import pytest
import tempfile
from src.db_manager import DatabaseManager

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    import gc
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass

def test_db_insert_and_fetch(temp_db):
    sample_features = {
        'CGPA': 8.5,
        'Programming_Score': 85.0,
        'Communication_Score': 75.0,
        'Aptitude_Score': 80.0,
        'Internships': 2,
        'Projects': 3,
        'Technical_Certifications': 2,
        'Work_Experience': 'Yes',
        'Backlogs': 0,
        'Extracurricular_Activities': 'Yes'
    }
    row_id = temp_db.log_prediction(
        student_identifier="STU_TEST_01",
        features=sample_features,
        prediction="Placed",
        probability=0.8842,
        confidence=88.42,
        threshold=0.5
    )
    assert row_id > 0
    
    records = temp_db.get_all_records()
    assert len(records) == 1
    assert records.iloc[0]['student_identifier'] == "STU_TEST_01"
    assert records.iloc[0]['prediction'] == "Placed"
    
    stats = temp_db.get_stats()
    assert stats['total_predictions'] == 1
    assert stats['placed_count'] == 1
    assert stats['avg_cgpa'] == 8.5
    
    temp_db.clear_all()
    assert len(temp_db.get_all_records()) == 0
