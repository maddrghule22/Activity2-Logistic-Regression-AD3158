"""
Smart Student Performance & Placement Prediction System
Preprocessor Module: Handles data cleaning, validation, encoding, scaling, and pipeline creation.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Default standard columns
DEFAULT_NUMERIC_FEATURES = [
    'CGPA', 'Programming_Score', 'Communication_Score', 'Aptitude_Score',
    'Internships', 'Projects', 'Technical_Certifications', 'Backlogs'
]
DEFAULT_CATEGORICAL_FEATURES = ['Work_Experience', 'Extracurricular_Activities']
DEFAULT_TARGET = 'Placement_Status'

class DataPreprocessor:
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_col: str = DEFAULT_TARGET,
        scaling_method: str = "StandardScaler"
    ):
        self.numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
        self.categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
        self.target_col = target_col
        self.scaling_method = scaling_method
        self.preprocessor_pipeline: Optional[ColumnTransformer] = None
        self.target_mapping = {'Not Placed': 0, 'Placed': 1, 0: 0, 1: 1}
        self.inv_target_mapping = {0: 'Not Placed', 1: 'Placed'}

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Returns structural dimensions, missing values, duplicates, and column types."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.round(2).values
        })
        
        return {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'duplicates': int(df.duplicated().sum()),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_summary': missing_df,
            'dtypes': df.dtypes.astype(str).to_dict()
        }

    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """Removes duplicates and cleans invalid records without data leakage."""
        cleaned_df = df.copy()
        if drop_duplicates:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            
        # Strip string whitespace across all string/object columns
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == object or pd.api.types.is_string_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
        return cleaned_df

    def build_transformer(self) -> ColumnTransformer:
        """Constructs an sklearn ColumnTransformer with imputation and scaling."""
        scaler = StandardScaler() if self.scaling_method == "StandardScaler" else MinMaxScaler()
        
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
        ])
        
        transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        self.preprocessor_pipeline = transformer
        return transformer

    def prepare_train_test_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
        drop_duplicates: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Cleans data and splits into train and test sets with stratification.
        Guarantees no data leakage.
        """
        clean_df = self.clean_data(df, drop_duplicates=drop_duplicates)
        
        # Verify target exists
        if self.target_col not in clean_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
            
        # Drop rows where target is NaN
        clean_df = clean_df.dropna(subset=[self.target_col])
        
        # Map target to binary numeric
        y_raw = clean_df[self.target_col]
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
        else:
            y = y_raw.astype(str).str.strip().map(self.target_mapping).fillna(0).astype(int)
            
        feature_cols = [c for c in (self.numeric_features + self.categorical_features) if c in clean_df.columns]
        X = clean_df[feature_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
