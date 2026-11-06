"""
Smart Student Performance & Placement Prediction System
Plot Generator Utility: Produces high-resolution publication-quality PNG charts for project report and presentation.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from src.data_generator import generate_student_dataset
from src.preprocessor import DataPreprocessor
from src.model_engine import ModelEngine
from src.evaluator import ModelEvaluator

os.makedirs("assets", exist_ok=True)
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['figure.dpi'] = 300

def generate_report_figures():
    df = pd.read_csv("data/student_placement_dataset.csv")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_train_test_data(df, test_size=0.20, random_state=42)
    transformer = preprocessor.build_transformer()
    
    engine = ModelEngine(preprocessor_pipeline=transformer)
    pipeline = engine.train_logistic_regression(X_train, y_train, C=1.0)
    
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
    plt.title("Logistic Regression Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png")
    plt.close()
    print("Saved assets/confusion_matrix.png")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    eval_res = ModelEvaluator.evaluate_predictions(y_test, y_proba)
    auc_val = eval_res['auc_score']
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f"Logistic Regression (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, linestyle='--', label="Random Guess (AUC = 0.500)")
    plt.fill_between(fpr, tpr, color='#2563EB', alpha=0.12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("assets/roc_curve.png")
    plt.close()
    print("Saved assets/roc_curve.png")

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color='#10B981', lw=2.5, label="Precision-Recall Curve")
    plt.fill_between(rec, prec, color='#10B981', alpha=0.12)
    plt.title("Precision-Recall Curve", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig("assets/pr_curve.png")
    plt.close()
    print("Saved assets/pr_curve.png")

    # 4. Coefficients & Feature Importance
    df_coef, intercept = engine.get_feature_coefficients()
    plt.figure(figsize=(8, 5))
    colors = ['#10B981' if b >= 0 else '#EF4444' for b in df_coef['Coefficient (Beta)']]
    sns.barplot(data=df_coef, x='Coefficient (Beta)', y='Feature', palette=colors)
    plt.title("Logistic Regression Feature Weights (Beta Coefficients)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Beta Coefficient Value (Log-Odds Impact)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png")
    plt.close()
    print("Saved assets/feature_importance.png")

    # 5. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
    plt.title("Dataset Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("assets/correlation_heatmap.png")
    plt.close()
    print("Saved assets/correlation_heatmap.png")

if __name__ == "__main__":
    generate_report_figures()
