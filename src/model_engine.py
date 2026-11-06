"""
Smart Student Performance & Placement Prediction System
Model Engine Module: Trains Logistic Regression, performs Hyperparameter Tuning, Cross-Validation,
and model comparisons (Logistic Regression, Decision Tree, Random Forest).
"""

import os
import warnings
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEngine:
    def __init__(self, preprocessor_pipeline=None):
        self.preprocessor_pipeline = preprocessor_pipeline
        self.trained_pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.best_params: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}

    def build_pipeline(self, classifier=None) -> Pipeline:
        """Combines preprocessor and estimator into a leakage-free Scikit-Learn Pipeline."""
        if classifier is None:
            classifier = LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor_pipeline),
            ('classifier', classifier)
        ])
        return pipeline

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        l1_ratio: Optional[float] = None,
        random_state: int = 42
    ) -> Pipeline:
        """Trains Logistic Regression with user-configured parameters."""
        lr_kwargs = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'random_state': random_state
        }
        if penalty == 'elasticnet' and solver == 'saga':
            lr_kwargs['l1_ratio'] = l1_ratio if l1_ratio is not None else 0.5
            
        clf = LogisticRegression(**lr_kwargs)
        self.trained_pipeline = self.build_pipeline(classifier=clf)
        self.trained_pipeline.fit(X_train, y_train)
        
        # Extract transformed feature names
        self._extract_feature_names(X_train)
        return self.trained_pipeline

    def perform_hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Runs GridSearchCV for hyperparameter optimization across C and penalties."""
        pipeline = self.build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        
        # Grid covering L1, L2, ElasticNet with appropriate solvers
        param_grid = [
            {
                'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            {
                'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0],
                'classifier__penalty': ['l1'],
                'classifier__solver': ['liblinear', 'saga']
            }
        ]
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.trained_pipeline = grid_search.best_estimator_
        self._extract_feature_names(X_train)
        
        # Format results into a dataframe
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cols_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        cv_summary = cv_results_df[cols_to_keep].sort_values(by='rank_test_score').reset_index(drop=True)
        
        return {
            'best_params': self.best_params,
            'best_f1_score': float(grid_search.best_score_),
            'cv_summary_table': cv_summary,
            'best_model': self.trained_pipeline
        }

    def evaluate_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Computes k-fold cross validation scores for multiple metrics."""
        if self.trained_pipeline is None:
            raise ValueError("Model pipeline has not been trained yet.")
            
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_validate(self.trained_pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        summary = {}
        for metric in scoring:
            test_vals = cv_scores[f'test_{metric}']
            train_vals = cv_scores[f'train_{metric}']
            summary[metric] = {
                'test_mean': float(np.mean(test_vals)),
                'test_std': float(np.std(test_vals)),
                'train_mean': float(np.mean(train_vals)),
                'train_std': float(np.std(train_vals)),
                'fold_scores': [float(s) for s in test_vals]
            }
        self.cv_results = summary
        return summary

    def get_feature_coefficients(self) -> pd.DataFrame:
        """Returns model weights, odds ratios, and relative importance."""
        if self.trained_pipeline is None:
            raise ValueError("Model has not been trained.")
            
        classifier = self.trained_pipeline.named_steps['classifier']
        if not hasattr(classifier, 'coef_'):
            raise ValueError("The estimator does not have linear coefficients.")
            
        coefs = classifier.coef_[0]
        intercept = classifier.intercept_[0]
        
        df_coef = pd.DataFrame({
            'Feature': self.feature_names if len(self.feature_names) == len(coefs) else [f"Feature_{i}" for i in range(len(coefs))],
            'Coefficient (Beta)': np.round(coefs, 4),
            'Odds_Ratio': np.round(np.exp(coefs), 4),
            'Absolute_Impact': np.round(np.abs(coefs), 4),
            'Impact_Direction': ['Positive (+)' if c > 0 else 'Negative (-)' for c in coefs]
        })
        df_coef = df_coef.sort_values(by='Absolute_Impact', ascending=False).reset_index(drop=True)
        return df_coef, intercept

    def compare_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
        """Trains Logistic Regression, Decision Tree, and Random Forest for empirical comparison."""
        models = {
            'Logistic Regression': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        }
        
        results = []
        fitted_pipelines = {}
        
        for name, clf in models.items():
            pipe = self.build_pipeline(classifier=clf)
            pipe.fit(X_train, y_train)
            fitted_pipelines[name] = pipe
            
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            
            results.append({
                'Model': name,
                'Accuracy': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'F1 Score': float(f1_score(y_test, y_pred, zero_division=0)),
                'ROC AUC': float(roc_auc_score(y_test, y_proba))
            })
            
        comparison_df = pd.DataFrame(results).round(4)
        return comparison_df, fitted_pipelines

    def _extract_feature_names(self, X_sample: pd.DataFrame):
        """Extracts transformed feature names from ColumnTransformer."""
        try:
            preprocessor = self.trained_pipeline.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                raw_names = preprocessor.get_feature_names_out()
                clean_names = [n.replace('num__', '').replace('cat__', '') for n in raw_names]
                self.feature_names = clean_names
            else:
                self.feature_names = list(X_sample.columns)
        except Exception:
            self.feature_names = list(X_sample.columns)

    def save_model(self, filepath: str = "models/placement_logistic_model.joblib") -> str:
        """Serializes the trained pipeline with joblib."""
        if self.trained_pipeline is None:
            raise ValueError("No trained model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'pipeline': self.trained_pipeline,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }, filepath)
        return filepath

    @staticmethod
    def load_model(filepath: str = "models/placement_logistic_model.joblib") -> Dict[str, Any]:
        """Loads serialized pipeline and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)
