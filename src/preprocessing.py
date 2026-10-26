import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.csv"
)

TARGET_COLUMN = "Churn"

# Numerical features (continuous)
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def load_train_test_data():
    """Load pre-split train and test datasets."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    return train_df, test_df


def separate_features_target(train_df, test_df):
    """Separate features from target in train and test sets."""
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def prepare_feature_types(X_train, X_test=None):
    """
    Prepare feature types for preprocessing:
    - Convert TotalCharges to numeric (handling string values)
    - Convert SeniorCitizen to string (treat as categorical)
    
    Rationale for TotalCharges imputation:
    - TotalCharges contains 11 missing values (0.16% of data)
    - All missing values correspond to customers with tenure=0 (new customers)
    - Zero imputation is domain-appropriate: new customers haven't been billed yet
    - This preserves the relationship between tenure=0 and TotalCharges=0
    """
    X_train = X_train.copy()
    
    # Convert TotalCharges to numeric (coerce errors to NaN for missing values)
    X_train["TotalCharges"] = pd.to_numeric(
        X_train["TotalCharges"],
        errors="coerce"
    )

    # SeniorCitizen is a binary categorical feature (0/1 -> "0"/"1")
    X_train["SeniorCitizen"] = X_train["SeniorCitizen"].astype(str)

    if X_test is not None:
        X_test = X_test.copy()
        X_test["TotalCharges"] = pd.to_numeric(
            X_test["TotalCharges"],
            errors="coerce"
        )
        X_test["SeniorCitizen"] = X_test["SeniorCitizen"].astype(str)
        return X_train, X_test

    return X_train


def get_feature_columns(X_train):
    """
    Identify numerical and categorical feature columns.
    
    Returns:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names (all non-numerical)
    """
    numerical_features = NUMERICAL_FEATURES.copy()
    
    categorical_features = [
        column
        for column in X_train.columns
        if column not in numerical_features
    ]

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Build an UNFITTED ColumnTransformer for preprocessing.
    
    This preprocessor can be fit on training data during cross-validation
    to avoid data leakage. The same unfitted preprocessor can be reused
    across different training/validation splits.
    
    Numerical pipeline:
    - Impute missing values with 0 (for TotalCharges tenure=0 cases)
    - StandardScaler for normalization
    
    Categorical pipeline:
    - Impute missing values with most frequent
    - OneHotEncoder with handle_unknown='ignore' for inference safety
    
    Args:
        numerical_features: List of numerical feature column names
        categorical_features: List of categorical feature column names
        
    Returns:
        Unfitted ColumnTransformer ready for fit_transform
    """
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(
                    strategy="constant",
                    fill_value=0
                )
            ),
            (
                "scaler", StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical", numerical_pipeline, numerical_features
            ),
            (
                "categorical", categorical_pipeline, categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


def get_preprocessor_and_features(X_train):
    """
    Convenience function to get preprocessor and feature lists.
    
    This prepares the data and returns an unfitted preprocessor
    along with the feature lists. The preprocessor is NOT fit here
    to avoid data leakage during cross-validation.
    
    Args:
        X_train: Training features DataFrame
        
    Returns:
        preprocessor: Unfitted ColumnTransformer
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        X_train_prepared: X_train with type conversions applied
    """
    X_train_prepared = prepare_feature_types(X_train)
    numerical_features, categorical_features = get_feature_columns(X_train_prepared)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    return preprocessor, numerical_features, categorical_features, X_train_prepared


if __name__ == "__main__":
    # Verification script - does NOT fit preprocessor on full training set
    train_df, test_df = load_train_test_data()
    X_train, X_test, y_train, y_test = separate_features_target(train_df, test_df)
    
    # Prepare feature types
    X_train_prepared, X_test_prepared = prepare_feature_types(X_train, X_test)
    
    # Get feature lists
    numerical_features, categorical_features = get_feature_columns(X_train_prepared)
    
    # Build unfitted preprocessor
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    print("\n========== PREPROCESSING SETUP VERIFICATION ==========")
    print(f"\nOriginal training rows: {len(y_train)}")
    print(f"Original test rows: {len(y_test)}")
    print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    print(f"\nTotalCharges missing in train: {X_train_prepared['TotalCharges'].isnull().sum()}")
    print(f"TotalCharges missing in test: {X_test_prepared['TotalCharges'].isnull().sum()}")
    
    print(f"\nSeniorCitizen dtype: {X_train_prepared['SeniorCitizen'].dtype}")
    
    print("\nPreprocessor is UNFITTED - ready for cross-validation")
    print("=======================================================")