"""
Tests for data loading, validation, and preprocessing pipeline.

These tests verify:
1. Target mapping (No=0, Yes=1)
2. Feature type identification (numerical vs categorical)
3. Missing value handling (TotalCharges)
4. Preprocessor construction (unfitted ColumnTransformer)
5. Data leakage prevention (preprocessor not fitted on full data)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data_loader import (
    load_data,
    validate_data,
    prepare_data,
    split_data,
    EXPECTED_TARGET_VALUES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN
)

from src.preprocessing import (
    prepare_feature_types,
    get_feature_columns,
    build_preprocessor,
    get_preprocessor_and_features,
    NUMERICAL_FEATURES
)


class TestDataLoader:
    """Test data loading and validation functions."""

    def test_load_data(self):
        """Test that raw data can be loaded."""
        df = load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_validate_data_with_valid_data(self):
        """Test validation passes with valid data."""
        df = load_data()
        # Should not raise any exception
        assert validate_data(df) is True

    def test_validate_data_missing_column(self):
        """Test validation fails with missing required column."""
        df = load_data()
        df_missing = df.drop(columns=["gender"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df_missing)

    def test_validate_data_missing_target_values(self):
        """Test validation fails with missing target values."""
        df = load_data()
        df.loc[0, TARGET_COLUMN] = np.nan
        with pytest.raises(ValueError, match="missing values in target column"):
            validate_data(df)

    def test_validate_data_invalid_target_value(self):
        """Test validation fails with unexpected target value."""
        df = load_data()
        df.loc[0, TARGET_COLUMN] = "Maybe"
        with pytest.raises(ValueError, match="unexpected target values"):
            validate_data(df)

    def test_prepare_data_target_mapping(self):
        """Test target encoding: No=0, Yes=1."""
        df = load_data()
        df_prepared = prepare_data(df)
        
        # Check target is binary
        assert set(df_prepared[TARGET_COLUMN].unique()) == {0, 1}
        
        # Verify mapping
        original_no_count = (df[TARGET_COLUMN] == "No").sum()
        original_yes_count = (df[TARGET_COLUMN] == "Yes").sum()
        
        encoded_zero_count = (df_prepared[TARGET_COLUMN] == 0).sum()
        encoded_one_count = (df_prepared[TARGET_COLUMN] == 1).sum()
        
        assert original_no_count == encoded_zero_count
        assert original_yes_count == encoded_one_count

    def test_prepare_data_drops_id_column(self):
        """Test that customerID column is dropped."""
        df = load_data()
        df_prepared = prepare_data(df)
        assert ID_COLUMN not in df_prepared.columns

    def test_split_data_stratified(self):
        """Test that train-test split is stratified."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, test_df = split_data(df_prepared)
        
        # Check split ratio is approximately 80:20
        total_len = len(df_prepared)
        train_ratio = len(train_df) / total_len
        test_ratio = len(test_df) / total_len
        
        assert 0.78 < train_ratio < 0.82
        assert 0.18 < test_ratio < 0.22
        
        # Check stratification preserves class distribution
        train_churn_rate = train_df[TARGET_COLUMN].mean()
        test_churn_rate = test_df[TARGET_COLUMN].mean()
        overall_churn_rate = df_prepared[TARGET_COLUMN].mean()
        
        # Allow small deviation due to rounding
        assert abs(train_churn_rate - overall_churn_rate) < 0.02
        assert abs(test_churn_rate - overall_churn_rate) < 0.02


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_prepare_feature_types_totalcharges_numeric(self):
        """Test TotalCharges is converted to numeric."""
        df = load_data()
        df_prepared = prepare_data(df)
        X = df_prepared.drop(columns=[TARGET_COLUMN])
        
        X_prepared = prepare_feature_types(X)
        
        assert pd.api.types.is_numeric_dtype(X_prepared["TotalCharges"])

    def test_prepare_feature_types_totalcharges_missing(self):
        """Test TotalCharges missing values are identified."""
        df = load_data()
        df_prepared = prepare_data(df)
        X = df_prepared.drop(columns=[TARGET_COLUMN])
        
        X_prepared = prepare_feature_types(X)
        
        # Should have 11 missing values (tenure=0 customers)
        missing_count = X_prepared["TotalCharges"].isnull().sum()
        assert missing_count == 11

    def test_prepare_feature_types_seniorcitizen_categorical(self):
        """Test SeniorCitizen is converted to string (categorical)."""
        df = load_data()
        df_prepared = prepare_data(df)
        X = df_prepared.drop(columns=[TARGET_COLUMN])
        
        X_prepared = prepare_feature_types(X)
        
        assert pd.api.types.is_string_dtype(X_prepared["SeniorCitizen"])
        assert set(X_prepared["SeniorCitizen"].unique()) == {"0", "1"}

    def test_prepare_feature_types_with_test_data(self):
        """Test prepare_feature_types works with both train and test."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, test_df = split_data(df_prepared)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep, X_test_prep = prepare_feature_types(X_train, X_test)
        
        assert pd.api.types.is_numeric_dtype(X_train_prep["TotalCharges"])
        assert pd.api.types.is_numeric_dtype(X_test_prep["TotalCharges"])
        assert pd.api.types.is_string_dtype(X_train_prep["SeniorCitizen"])
        assert pd.api.types.is_string_dtype(X_test_prep["SeniorCitizen"])

    def test_get_feature_columns_identifies_numerical(self):
        """Test numerical features are correctly identified."""
        df = load_data()
        df_prepared = prepare_data(df)
        X = df_prepared.drop(columns=[TARGET_COLUMN])
        X_prepared = prepare_feature_types(X)
        
        numerical_features, categorical_features = get_feature_columns(X_prepared)
        
        assert numerical_features == NUMERICAL_FEATURES
        assert len(numerical_features) == 3
        assert "tenure" in numerical_features
        assert "MonthlyCharges" in numerical_features
        assert "TotalCharges" in numerical_features

    def test_get_feature_columns_identifies_categorical(self):
        """Test categorical features are correctly identified."""
        df = load_data()
        df_prepared = prepare_data(df)
        X = df_prepared.drop(columns=[TARGET_COLUMN])
        X_prepared = prepare_feature_types(X)
        
        numerical_features, categorical_features = get_feature_columns(X_prepared)
        
        # SeniorCitizen should be in categorical
        assert "SeniorCitizen" in categorical_features
        # Numerical features should not be in categorical
        assert "tenure" not in categorical_features
        assert "MonthlyCharges" not in categorical_features
        assert "TotalCharges" not in categorical_features

    def test_build_preprocessor_returns_unfitted(self):
        """Test that build_preprocessor returns an unfitted transformer."""
        numerical_features = NUMERICAL_FEATURES
        categorical_features = ["gender", "SeniorCitizen"]
        
        preprocessor = build_preprocessor(numerical_features, categorical_features)
        
        # Check it's a ColumnTransformer
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)
        
        # Check it's unfitted (no attributes learned yet)
        assert not hasattr(preprocessor, "transformers_")

    def test_build_preprocessor_structure(self):
        """Test preprocessor has correct pipeline structure."""
        numerical_features = NUMERICAL_FEATURES
        categorical_features = ["gender", "SeniorCitizen"]
        
        preprocessor = build_preprocessor(numerical_features, categorical_features)
        
        # Check transformers
        assert len(preprocessor.transformers) == 2
        assert preprocessor.transformers[0][0] == "numerical"
        assert preprocessor.transformers[1][0] == "categorical"

    def test_get_preprocessor_and_features_convenience(self):
        """Test convenience function returns correct components."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, test_df = split_data(df_prepared)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        
        preprocessor, num_features, cat_features, X_train_prep = (
            get_preprocessor_and_features(X_train)
        )
        
        # Check all components are returned
        assert preprocessor is not None
        assert num_features == NUMERICAL_FEATURES
        assert len(cat_features) > 0
        assert X_train_prep is not None
        
        # Check preprocessor is unfitted
        assert not hasattr(preprocessor, "transformers_")

    def test_preprocessor_can_be_fit(self):
        """Test that preprocessor can be fit on data."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, test_df = split_data(df_prepared)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        
        preprocessor, num_features, cat_features, X_train_prep = (
            get_preprocessor_and_features(X_train)
        )
        
        # Fit on training data
        X_train_transformed = preprocessor.fit_transform(X_train_prep)
        
        # Check transformation worked
        assert X_train_transformed.shape[0] == len(X_train_prep)
        assert X_train_transformed.shape[1] > len(num_features) + len(cat_features)
        
        # Check no missing values after transformation
        assert not np.isnan(X_train_transformed).any()

    def test_preprocessor_can_transform_test_data(self):
        """Test that preprocessor can transform test data after fitting."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, test_df = split_data(df_prepared)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep, X_test_prep = prepare_feature_types(X_train, X_test)
        
        preprocessor, num_features, cat_features, _ = (
            get_preprocessor_and_features(X_train)
        )
        
        # Fit on train, transform both
        X_train_transformed = preprocessor.fit_transform(X_train_prep)
        X_test_transformed = preprocessor.transform(X_test_prep)
        
        # Check shapes match
        assert X_train_transformed.shape[0] == len(X_train_prep)
        assert X_test_transformed.shape[0] == len(X_test_prep)
        
        # Check same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        
        # Check no missing values
        assert not np.isnan(X_test_transformed).any()


class TestDataLeakagePrevention:
    """Test that preprocessing pipeline prevents data leakage."""

    def test_preprocessor_not_fitted_in_convenience_function(self):
        """Test that get_preprocessor_and_features does not fit preprocessor."""
        df = load_data()
        df_prepared = prepare_data(df)
        train_df, _ = split_data(df_prepared)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        # Preprocessor should be unfitted
        assert not hasattr(preprocessor, "transformers_")
        assert not hasattr(preprocessor, "named_transformers_")

    def test_preprocessing_module_main_does_not_fit(self):
        """Test that preprocessing.py __main__ does not fit on full data."""
        # Import and run the verification script
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "src.preprocessing"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        
        # Should complete successfully
        assert result.returncode == 0
        
        # Should mention unfitted preprocessor
        assert "UNFITTED" in result.stdout
