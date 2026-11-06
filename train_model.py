"""
Smart Student Performance & Placement Prediction System
Standalone Model Training Script
"""

import os
import pandas as pd
from src.data_generator import generate_student_dataset
from src.preprocessor import DataPreprocessor
from src.model_engine import ModelEngine
from src.evaluator import ModelEvaluator

def run_training_pipeline():
    dataset_path = "data/student_placement_dataset.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating default student placement dataset...")
        df = generate_student_dataset(n_samples=1200, random_state=42)
        os.makedirs("data", exist_ok=True)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 1. Preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_train_test_data(df, test_size=0.20, random_state=42)
    transformer = preprocessor.build_transformer()
    
    # 2. Model Training
    engine = ModelEngine(preprocessor_pipeline=transformer)
    pipeline = engine.train_logistic_regression(
        X_train, y_train,
        C=1.0,
        penalty='l2',
        solver='lbfgs'
    )
    
    # 3. Model Evaluation
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    eval_results = ModelEvaluator.evaluate_predictions(y_test, y_proba, threshold=0.50)
    
    print("\n================ MODEL TRAINING COMPLETED ================")
    print(f"Accuracy Score:   {eval_results['accuracy']:.4f}")
    print(f"Precision Score:  {eval_results['precision']:.4f}")
    print(f"Recall Score:     {eval_results['recall']:.4f}")
    print(f"F1 Score:         {eval_results['f1_score']:.4f}")
    print(f"ROC-AUC Score:    {eval_results['auc_score']:.4f}")
    print(f"Log Loss:         {eval_results['log_loss']:.4f}")
    print("---------------------------------------------------------")
    print("Confusion Matrix:")
    print(eval_results['confusion_matrix']['matrix'])
    print("=========================================================\n")
    
    # 4. Save Model
    saved_path = engine.save_model("models/placement_logistic_model.joblib")
    print(f"Model saved successfully to: {saved_path}")
    
    return pipeline, eval_results

if __name__ == "__main__":
    run_training_pipeline()
