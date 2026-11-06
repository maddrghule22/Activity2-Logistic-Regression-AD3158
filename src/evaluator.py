"""
Smart Student Performance & Placement Prediction System
Evaluator Module: Calculates metrics, confusion matrices, ROC/PR curves, dynamic threshold analytics,
and 2D decision boundary meshgrids.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss
)

class ModelEvaluator:
    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Calculates comprehensive classification metrics for given probabilities and threshold."""
        y_true = np.array(y_true).astype(int)
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc = roc_auc_score(y_true, y_proba)
        loss = log_loss(y_true, y_proba)
        
        # Classification report dict
        report_dict = classification_report(
            y_true, y_pred,
            target_names=['Not Placed', 'Placed'],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        
        # ROC Curve points
        fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
        
        # Precision-Recall Curve points
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_proba)
        
        return {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'auc_score': float(auc),
            'log_loss': float(loss),
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'classification_report_df': report_df,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresh.tolist()
            },
            'pr_curve': {
                'precision': prec.tolist(),
                'recall': rec.tolist(),
                'thresholds': pr_thresh.tolist()
            }
        }

    @staticmethod
    def compute_threshold_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        steps: int = 50
    ) -> pd.DataFrame:
        """Computes Precision, Recall, and F1 across multiple threshold levels (0.05 to 0.95)."""
        thresholds = np.linspace(0.05, 0.95, steps)
        records = []
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            records.append({
                'Threshold': round(th, 3),
                'Accuracy': round(accuracy_score(y_true, y_pred), 4),
                'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
                'F1 Score': round(f1_score(y_true, y_pred, zero_division=0), 4)
            })
        return pd.DataFrame(records)

    @staticmethod
    def generate_decision_boundary_2d(
        pipeline,
        X_df: pd.DataFrame,
        feat_x: str,
        feat_y: str,
        grid_resolution: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a 2D meshgrid of probabilities for two selected features
        while holding all other continuous features at median and categorical at mode.
        """
        x_min, x_max = X_df[feat_x].min(), X_df[feat_x].max()
        y_min, y_max = X_df[feat_y].min(), X_df[feat_y].max()
        
        x_padding = (x_max - x_min) * 0.08
        y_padding = (y_max - y_min) * 0.08
        
        xx, yy = np.meshgrid(
            np.linspace(x_min - x_padding, x_max + x_padding, grid_resolution),
            np.linspace(y_min - y_padding, y_max + y_padding, grid_resolution)
        )
        
        # Build baseline synthetic dataframe
        baseline = {}
        for col in X_df.columns:
            if col in [feat_x, feat_y]:
                continue
            if pd.api.types.is_numeric_dtype(X_df[col]):
                baseline[col] = X_df[col].median()
            else:
                baseline[col] = X_df[col].mode()[0] if not X_df[col].empty else 'No'
                
        grid_df = pd.DataFrame(baseline, index=range(xx.size))
        grid_df[feat_x] = xx.ravel()
        grid_df[feat_y] = yy.ravel()
        
        # Ensure column order matches X_df
        grid_df = grid_df[X_df.columns]
        
        # Predict probability
        probs = pipeline.predict_proba(grid_df)[:, 1]
        zz = probs.reshape(xx.shape)
        
        return xx, yy, zz
