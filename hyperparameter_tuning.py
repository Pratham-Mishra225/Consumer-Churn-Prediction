import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import get_preprocessor_and_features, prepare_feature_types

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
N_FOLDS = 5

def get_cv_splitter():
    """Return stratified K-fold splitter."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def get_scoring():
    """Return multi-metric scoring dictionary."""
    return {
        'avg_precision': 'average_precision',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

def compute_metrics_on_best(X, y, best_pipeline, cv):
    """Compute all metrics on best estimator using cross_validate."""
    scoring = {
        'avg_precision': 'average_precision',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        best_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    metrics = {}
    for metric in ['avg_precision', 'roc_auc', 'precision', 'recall', 'f1']:
        test_key = f'test_{metric}'
        metrics[f'cv_{metric}'] = cv_results[test_key].mean()
        metrics[f'cv_{metric}_std'] = cv_results[test_key].std()
    
    return metrics

def get_logistic_regression_param_grid():
    """Parameter grid for Logistic Regression."""
    return {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__class_weight': [None, 'balanced']
    }

def get_decision_tree_param_grid():
    """Parameter grid for Decision Tree."""
    return {
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 5, 10],
        'classifier__class_weight': [None, 'balanced']
    }

def get_random_forest_param_grid():
    """Parameter grid for Random Forest."""
    return {
        'classifier__n_estimators': [200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [1, 5],
        'classifier__max_features': ['sqrt', 0.5],
        'classifier__class_weight': [None, 'balanced']
    }

def get_svc_param_grid():
    """Parameter grid for SVC."""
    return [
        {
            'classifier__estimator__kernel': ['linear'],
            'classifier__estimator__C': [0.1, 1, 10],
            'classifier__estimator__class_weight': [None, 'balanced']
        },
        {
            'classifier__estimator__kernel': ['rbf'],
            'classifier__estimator__C': [0.1, 1, 10],
            'classifier__estimator__gamma': ['scale', 'auto'],
            'classifier__estimator__class_weight': [None, 'balanced']
        }
    ]

def get_base_model(model_name):
    """Return base model for tuning."""
    if model_name == 'Logistic Regression':
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == 'Support Vector Classifier':
        base_svc = SVC(random_state=42)
        return CalibratedClassifierCV(base_svc, ensemble=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_param_grid(model_name):
    """Return parameter grid for model."""
    if model_name == 'Logistic Regression':
        return get_logistic_regression_param_grid()
    elif model_name == 'Decision Tree':
        return get_decision_tree_param_grid()
    elif model_name == 'Random Forest':
        return get_random_forest_param_grid()
    elif model_name == 'Support Vector Classifier':
        return get_svc_param_grid()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_grid_search(X, y, preprocessor, model_name):
    """Run GridSearchCV for a single model."""
    print(f"\n{'='*80}")
    print(f"Tuning {model_name}")
    print(f"{'='*80}")
    
    base_model = get_base_model(model_name)
    param_grid = get_param_grid(model_name)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])
    
    cv = get_cv_splitter()
    scoring = get_scoring()
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='avg_precision',
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    
    start_time = time.time()
    grid_search.fit(X, y)
    elapsed_time = time.time() - start_time
    
    results = {
        'model': model_name,
        'best_params': grid_search.best_params_,
        'best_avg_precision': grid_search.best_score_,
        'fit_time_seconds': elapsed_time,
        'n_candidates': len(grid_search.cv_results_['params'])
    }
    
    # Compute all metrics on best estimator
    best_metrics = compute_metrics_on_best(X, y, grid_search.best_estimator_, cv)
    results.update(best_metrics)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best Avg Precision: {grid_search.best_score_:.4f}")
    print(f"CV ROC-AUC: {results['cv_roc_auc']:.4f}")
    print(f"CV F1-Score: {results['cv_f1']:.4f}")
    print(f"Fit time: {elapsed_time:.2f} seconds")
    print(f"Candidates evaluated: {results['n_candidates']}")
    
    return results

def save_results(all_results):
    """Save tuning results to CSV and JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    json_results = []
    for result in all_results:
        json_result = result.copy()
        json_result['best_params'] = str(json_result['best_params'])
        json_results.append(json_result)
    
    json_path = REPORTS_DIR / "hyperparameter_tuning_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nDetailed results saved to {json_path}")
    
    # Save summary as CSV
    summary_data = []
    for result in all_results:
        summary = {
            'model': result['model'],
            'best_avg_precision': result['best_avg_precision'],
            'cv_roc_auc': result['cv_roc_auc'],
            'cv_roc_auc_std': result['cv_roc_auc_std'],
            'cv_precision': result['cv_precision'],
            'cv_precision_std': result['cv_precision_std'],
            'cv_recall': result['cv_recall'],
            'cv_recall_std': result['cv_recall_std'],
            'cv_f1': result['cv_f1'],
            'cv_f1_std': result['cv_f1_std'],
            'fit_time_seconds': result['fit_time_seconds'],
            'n_candidates': result['n_candidates'],
            'best_params': str(result['best_params'])
        }
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = REPORTS_DIR / "hyperparameter_tuning_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to {csv_path}")
    
    return summary_df

def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    X = train_df.drop(columns=['Churn'])
    y = train_df['Churn']
    
    print("Preparing features...")
    X_prepared = prepare_feature_types(X)
    preprocessor, num_features, cat_features, _ = get_preprocessor_and_features(X)
    
    print(f"Numerical features: {len(num_features)}")
    print(f"Categorical features: {len(cat_features)}")
    
    models = [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'Support Vector Classifier'
    ]
    
    all_results = []
    
    for model_name in models:
        try:
            results = run_grid_search(X_prepared, y, preprocessor, model_name)
            all_results.append(results)
        except Exception as e:
            print(f"\nError tuning {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo results to save - all models failed tuning")
        return
    
    summary_df = save_results(all_results)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Identify best model by average precision
    if not summary_df.empty and 'best_avg_precision' in summary_df.columns:
        best_idx = summary_df['best_avg_precision'].idxmax()
        best_model = summary_df.loc[best_idx, 'model']
        best_score = summary_df.loc[best_idx, 'best_avg_precision']
        print(f"\nBest model by Average Precision: {best_model} ({best_score:.4f})")
        print(f"Best parameters: {summary_df.loc[best_idx, 'best_params']}")

if __name__ == "__main__":
    main()
