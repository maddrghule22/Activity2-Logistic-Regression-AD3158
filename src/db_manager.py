"""
Smart Student Performance & Placement Prediction System
Database Manager Module: SQLite persistence for student prediction records and audit history.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "prediction_history.db")

class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes prediction history schema if not already created."""
        query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            student_identifier TEXT NOT NULL,
            cgpa REAL NOT NULL,
            programming_score REAL NOT NULL,
            communication_score REAL NOT NULL,
            aptitude_score REAL NOT NULL,
            internships INTEGER NOT NULL,
            projects INTEGER NOT NULL,
            technical_certifications INTEGER NOT NULL,
            work_experience TEXT NOT NULL,
            backlogs INTEGER NOT NULL,
            extracurricular TEXT NOT NULL,
            prediction TEXT NOT NULL,
            placement_probability REAL NOT NULL,
            confidence_score REAL NOT NULL,
            classification_threshold REAL NOT NULL
        );
        """
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()

    def log_prediction(
        self,
        student_identifier: str,
        features: Dict[str, Any],
        prediction: str,
        probability: float,
        confidence: float,
        threshold: float
    ) -> int:
        """Stores a new student prediction query."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO predictions (
            timestamp, student_identifier, cgpa, programming_score,
            communication_score, aptitude_score, internships, projects,
            technical_certifications, work_experience, backlogs,
            extracurricular, prediction, placement_probability,
            confidence_score, classification_threshold
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (
                now,
                student_identifier,
                float(features.get('CGPA', 0.0)),
                float(features.get('Programming_Score', 0.0)),
                float(features.get('Communication_Score', 0.0)),
                float(features.get('Aptitude_Score', 0.0)),
                int(features.get('Internships', 0)),
                int(features.get('Projects', 0)),
                int(features.get('Technical_Certifications', 0)),
                str(features.get('Work_Experience', 'No')),
                int(features.get('Backlogs', 0)),
                str(features.get('Extracurricular_Activities', 'No')),
                prediction,
                round(float(probability), 4),
                round(float(confidence), 2),
                round(float(threshold), 3)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_records(self) -> pd.DataFrame:
        """Retrieves all historical predictions as a Pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        return df

    def clear_all(self):
        """Clears all records in the prediction table."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM predictions")
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates aggregated metrics for logged predictions."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if total == 0:
                return {
                    'total_predictions': 0,
                    'placed_count': 0,
                    'not_placed_count': 0,
                    'placed_rate': 0.0,
                    'avg_cgpa': 0.0
                }
            placed = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Placed'").fetchone()[0]
            avg_cgpa = conn.execute("SELECT AVG(cgpa) FROM predictions").fetchone()[0]
            
            return {
                'total_predictions': total,
                'placed_count': placed,
                'not_placed_count': total - placed,
                'placed_rate': round((placed / total) * 100, 1),
                'avg_cgpa': round(avg_cgpa, 2) if avg_cgpa else 0.0
            }
