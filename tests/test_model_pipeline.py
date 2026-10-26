"""
Tests for model training, prediction, and end-to-end pipeline.

These tests verify:
1. Preprocessor compatibility with all candidate classifiers
2. Unknown categorical values during inference
3. Model training and prediction without errors
4. Valid prediction probabilities between 0 and 1
5. Consistent feature columns and input schema
6. Saved pipeline loading and prediction
7. Stable training configuration with fixed random seeds
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_data, prepare_data, split_data, TARGET_COLUMN
from preprocessing import (
    prepare_feature_types,
    get_preprocessor_and_features,
    build_preprocessor
)
from algorithms.model_registry import get_all_models
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RANDOM_STATE = 42


class TestPreprocessorCompatibility:
    """Test preprocessor compatibility with all candidate classifiers."""

    def test_preprocessor_with_logistic_regression(self):
        """Test preprocessor works with Logistic Regression."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        X_transformed = pipeline.predict_proba(X_train_prep)
        
        assert X_transformed.shape[0] == len(X_train_prep)
        assert X_transformed.shape[1] == 2

    def test_preprocessor_with_decision_tree(self):
        """Test preprocessor works with Decision Tree."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        predictions = pipeline.predict(X_train_prep)
        
        assert len(predictions) == len(X_train_prep)

    def test_preprocessor_with_random_forest(self):
        """Test preprocessor works with Random Forest."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=1))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        predictions = pipeline.predict(X_train_prep)
        
        assert len(predictions) == len(X_train_prep)

    def test_preprocessor_with_svc(self):
        """Test preprocessor works with SVC."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        base_svc = SVC(random_state=42)
        calibrated_svc = CalibratedClassifierCV(base_svc, ensemble=False)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', calibrated_svc)
        ])
        
        pipeline.fit(X_train_prep, y_train)
        X_transformed = pipeline.predict_proba(X_train_prep)
        
        assert X_transformed.shape[0] == len(X_train_prep)
        assert X_transformed.shape[1] == 2


class TestUnknownCategoricalValues:
    """Test handling of unknown categorical values during inference."""

    def test_unknown_categorical_value_handling(self):
        """Test pipeline handles unknown categorical values gracefully."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep = prepare_feature_types(X_train)
        X_test_prep = prepare_feature_types(X_test)
        
        # Introduce unknown categorical value in test set
        X_test_modified = X_test_prep.copy()
        X_test_modified.loc[0, 'InternetService'] = 'UnknownProvider'
        
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        
        # Should not raise error on unknown categorical value
        predictions = pipeline.predict(X_test_modified)
        
        assert len(predictions) == len(X_test_modified)


class TestModelTrainingAndPrediction:
    """Test model training and prediction functionality."""

    def test_model_training_without_errors(self):
        """Test all models can train without errors."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        model_registry = get_all_models()
        
        for model_name, model_factory in model_registry.items():
            model = model_factory()
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train_prep, y_train)
            predictions = pipeline.predict(X_train_prep)
            
            assert len(predictions) == len(X_train_prep)

    def test_prediction_probabilities_valid_range(self):
        """Test prediction probabilities are between 0 and 1."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep = prepare_feature_types(X_train)
        X_test_prep = prepare_feature_types(X_test)
        
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        proba = pipeline.predict_proba(X_test_prep)
        
        # Check probabilities are in valid range
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_prediction_consistency(self):
        """Test predictions are consistent with probabilities."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep = prepare_feature_types(X_train)
        X_test_prep = prepare_feature_types(X_test)
        
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        proba = pipeline.predict_proba(X_test_prep)
        predictions = pipeline.predict(X_test_prep)
        
        # Check predictions match argmax of probabilities
        predicted_from_proba = np.argmax(proba, axis=1)
        assert np.array_equal(predictions, predicted_from_proba)


class TestFeatureConsistency:
    """Test consistent feature columns and input schema."""

    def test_train_test_feature_columns_match(self):
        """Test train and test have same feature columns."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        assert set(X_train.columns) == set(X_test.columns)

    def test_preprocessor_output_consistency(self):
        """Test preprocessor outputs consistent feature dimensions."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_prep = prepare_feature_types(X_train)
        X_test_prep = prepare_feature_types(X_test)
        
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train_prep, y_train)
        
        X_train_transformed = pipeline[:-1].transform(X_train_prep)
        X_test_transformed = pipeline[:-1].transform(X_test_prep)
        
        # Check same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]


class TestSavedPipeline:
    """Test saved pipeline loading and prediction."""

    def test_saved_pipeline_loading(self):
        """Test saved pipeline can be loaded and used for prediction."""
        if not (SAVED_MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Saved model not found - run final_evaluation.py first")
        
        test_df = pd.read_csv(TEST_DATA_PATH)
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        
        X_test_prep = prepare_feature_types(X_test)
        
        # Load saved pipeline
        pipeline = joblib.load(SAVED_MODELS_DIR / "best_model.joblib")
        
        # Make predictions
        predictions = pipeline.predict(X_test_prep)
        proba = pipeline.predict_proba(X_test_prep)
        
        assert len(predictions) == len(X_test_prep)
        assert proba.shape[0] == len(X_test_prep)
        assert proba.shape[1] == 2

    def test_saved_pipeline_metadata(self):
        """Test saved metadata is valid JSON."""
        if not (SAVED_MODELS_DIR / "model_metadata.json").exists():
            pytest.skip("Model metadata not found - run final_evaluation.py first")
        
        import json
        with open(SAVED_MODELS_DIR / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Check required fields
        assert 'model_name' in metadata
        assert 'best_params' in metadata
        assert 'operating_threshold' in metadata
        assert 'cv_metrics' in metadata
        assert 'test_metrics' in metadata
        
        # Check threshold is valid
        assert 0 <= metadata['operating_threshold'] <= 1


class TestReproducibility:
    """Test stable training configuration with fixed random seeds."""

    def test_random_state_produces_same_results(self):
        """Test fixed random state produces reproducible results."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        # Train model twice with same random state
        pipeline1 = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline2 = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline1.fit(X_train_prep, y_train)
        pipeline2.fit(X_train_prep, y_train)
        
        # Predictions should be identical
        pred1 = pipeline1.predict(X_train_prep)
        pred2 = pipeline2.predict(X_train_prep)
        
        assert np.array_equal(pred1, pred2)

    def test_different_random_states_produce_different_results(self):
        """Test different random states produce different results."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_train_prep = prepare_feature_types(X_train)
        preprocessor1, _, _, _ = get_preprocessor_and_features(X_train)
        preprocessor2, _, _, _ = get_preprocessor_and_features(X_train)
        
        # Use Random Forest which is more sensitive to random state
        pipeline1 = Pipeline([
            ('preprocessor', preprocessor1),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=1))
        ])
        
        pipeline2 = Pipeline([
            ('preprocessor', preprocessor2),
            ('classifier', RandomForestClassifier(random_state=123, n_estimators=10, n_jobs=1))
        ])
        
        pipeline1.fit(X_train_prep, y_train)
        pipeline2.fit(X_train_prep, y_train)
        
        # Predictions should differ for ensemble models with different random states
        pred1 = pipeline1.predict(X_train_prep)
        pred2 = pipeline2.predict(X_train_prep)
        
        # At least some predictions should differ
        assert not np.array_equal(pred1, pred2)


class TestTargetLeakagePrevention:
    """Test no target leakage into model features."""

    def test_target_column_not_in_features(self):
        """Test target column is not in feature set."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        
        assert TARGET_COLUMN not in X_train.columns

    def test_feature_columns_do_not_contain_target_info(self):
        """Test feature columns don't contain target information."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        
        # Check no column contains 'Churn' in name
        for col in X_train.columns:
            assert 'Churn' not in col.lower()


class TestIntegration:
    """Integration test for lightweight training path."""

    def test_lightweight_training_path(self):
        """Test complete training path without expensive grid search."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        
        # Prepare features
        X_train_prep = prepare_feature_types(X_train)
        X_test_prep = prepare_feature_types(X_test)
        
        # Build preprocessor
        preprocessor, _, _, _ = get_preprocessor_and_features(X_train)
        
        # Train simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train_prep, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test_prep)
        y_proba = pipeline.predict_proba(X_test_prep)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        # Check metrics are reasonable
        assert 0 <= roc_auc <= 1
        assert 0 <= avg_precision <= 1
        assert roc_auc > 0.5  # Better than random
        assert avg_precision > 0.2  # Better than random for imbalanced data
        
        # Check predictions are valid
        assert set(y_pred).issubset({0, 1})
        assert len(y_pred) == len(y_test)
