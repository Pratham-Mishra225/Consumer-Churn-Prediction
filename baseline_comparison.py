import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    make_scorer
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import get_preprocessor_and_features, prepare_feature_types
from algorithms.model_registry import get_all_models

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
N_FOLDS = 5

def get_scoring_metrics():
    """Return scoring dictionary for cross-validation."""
    return {
        'avg_precision': 'average_precision',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'balanced_accuracy': 'balanced_accuracy'
    }

def build_pipeline(preprocessor, model):
    """Build pipeline with preprocessor and classifier."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def run_cross_validation(X, y, preprocessor, model_factory, cv):
    """Run cross-validation for a single model."""
    model = model_factory()
    pipeline = build_pipeline(preprocessor, model)
    scoring = get_scoring_metrics()
    
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    return results

def aggregate_results(results):
    """Aggregate cross-validation results to mean and std."""
    aggregated = {}
    for metric in ['avg_precision', 'roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy']:
        test_key = f'test_{metric}'
        aggregated[f'{metric}_mean'] = results[test_key].mean()
        aggregated[f'{metric}_std'] = results[test_key].std()
    return aggregated

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    X = train_df.drop(columns=['Churn'])
    y = train_df['Churn']
    
    print("Preparing features...")
    X_prepared = prepare_feature_types(X)
    preprocessor, num_features, cat_features, _ = get_preprocessor_and_features(X)
    
    print(f"Numerical features: {len(num_features)}")
    print(f"Categorical features: {len(cat_features)}")
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Using {N_FOLDS}-fold stratified cross-validation")
    
    model_registry = get_all_models()
    
    all_results = []
    
    for model_name, model_factory in model_registry.items():
        print(f"\nEvaluating {model_name}...")
        cv_results = run_cross_validation(X_prepared, y, preprocessor, model_factory, cv)
        aggregated = aggregate_results(cv_results)
        aggregated['model'] = model_name
        all_results.append(aggregated)
        
        print(f"  Avg Precision: {aggregated['avg_precision_mean']:.4f} (+/- {aggregated['avg_precision_std']:.4f})")
        print(f"  ROC-AUC: {aggregated['roc_auc_mean']:.4f} (+/- {aggregated['roc_auc_std']:.4f})")
        print(f"  F1-Score: {aggregated['f1_mean']:.4f} (+/- {aggregated['f1_std']:.4f})")
    
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    metric_cols = [col for col in results_df.columns if col != 'model']
    metric_cols_sorted = sorted(metric_cols, key=lambda x: x.split('_')[0])
    results_df = results_df[['model'] + metric_cols_sorted]
    
    output_path = REPORTS_DIR / "baseline_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Identify best model by average precision
    valid_results = results_df.dropna(subset=['avg_precision_mean'])
    if not valid_results.empty:
        best_model_idx = valid_results['avg_precision_mean'].idxmax()
        best_model = results_df.loc[best_model_idx, 'model']
        best_score = results_df.loc[best_model_idx, 'avg_precision_mean']
        print(f"\nBest model by Average Precision: {best_model} ({best_score:.4f})")
    else:
        print("\nCould not determine best model - all metrics are NaN")

if __name__ == "__main__":
    main()
