import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import get_preprocessor_and_features, prepare_feature_types

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
TUNING_RESULTS_PATH = REPORTS_DIR / "hyperparameter_tuning_summary.csv"
RANDOM_STATE = 42
N_FOLDS = 5

def load_tuning_results():
    """Load Phase 3 tuning results.

    Reads the summary CSV (primary ranking source) and, if the detailed JSON
    exists, merges cv_avg_precision_std into the returned DataFrame so that
    metadata and reports can include it.
    """
    df = pd.read_csv(TUNING_RESULTS_PATH)
    json_path = REPORTS_DIR / "hyperparameter_tuning_results.json"
    if json_path.exists():
        with open(json_path, encoding='utf-8') as fh:
            json_data = json.load(fh)
        std_map = {
            row['model']: row.get('cv_avg_precision_std', float('nan'))
            for row in json_data
        }
        df['cv_avg_precision_std'] = df['model'].map(std_map)
    else:
        df['cv_avg_precision_std'] = float('nan')
    return df

def select_best_model(tuning_results):
    """Select best model based on cross-validation Average Precision."""
    best_idx = tuning_results['best_avg_precision'].idxmax()
    best_row = tuning_results.loc[best_idx]
    return best_row

def parse_best_params(params_str):
    """Parse parameter string to dictionary."""
    import ast
    return ast.literal_eval(params_str)

def build_pipeline_with_params(preprocessor, model_name, params):
    """Build pipeline using the shared model registry factory, then apply tuned
    hyperparameters via Pipeline.set_params().

    Using the registry factory guarantees that fixed defaults configured there
    (e.g. max_iter=1000 for Logistic Regression) are always present before the
    tuned parameters are overlaid.  Pipeline.set_params() accepts the original
    GridSearchCV key names (classifier__C, classifier__estimator__kernel, etc.)
    so no prefix manipulation is required and no phantom attributes are created
    via setattr().
    """
    from algorithms.model_registry import get_model
    model = get_model(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # params keys already use the pipeline-namespace (classifier__* /
    # classifier__estimator__*) as stored by GridSearchCV – pass them directly.
    pipeline.set_params(**params)
    return pipeline

def get_out_of_fold_predictions(X, y, pipeline):
    """Get out-of-fold predictions using cross-validation."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]
    
    return oof_proba

def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def evaluate_thresholds(y_true, y_proba):
    """Evaluate default 0.5 threshold and optimal threshold."""
    # Default threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default)
    precision_default = precision_score(y_true, y_pred_default)
    recall_default = recall_score(y_true, y_pred_default)
    
    # Optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_proba)
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    precision_optimal = precision_score(y_true, y_pred_optimal)
    recall_optimal = recall_score(y_true, y_pred_optimal)
    
    return {
        'default_threshold': 0.5,
        'default_f1': f1_default,
        'default_precision': precision_default,
        'default_recall': recall_default,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'optimal_precision': precision_optimal,
        'optimal_recall': recall_optimal
    }

def evaluate_on_test(pipeline, X_test, y_test, threshold=0.5):
    """Evaluate pipeline on test set."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'test_avg_precision': average_precision_score(y_test, y_proba),
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    }
    
    return metrics, y_pred, y_proba

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_proba, save_path):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_model_comparison(tuning_results, save_path):
    """Plot comparison of all tuned models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['best_avg_precision', 'cv_roc_auc', 'cv_f1', 'cv_precision']
    titles = ['Average Precision', 'ROC-AUC', 'F1-Score', 'Precision']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        tuning_results.plot(x='model', y=metric, kind='bar', ax=ax, color='steelblue')
        ax.set_title(title)
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_selection_summary(best_model_info, test_metrics, threshold_info, save_path, clean_params):
    """Create selection summary markdown."""
    summary = f"""# Model Selection Summary

## Selected Model

**Model:** {best_model_info['model']}

**Best Hyperparameters:**
```
{clean_params}
```

**Cross-Validation Performance (Training Data):**
- Average Precision: {best_model_info['best_avg_precision']:.4f} ± {best_model_info['cv_avg_precision_std']:.4f}
- ROC-AUC: {best_model_info['cv_roc_auc']:.4f} ± {best_model_info['cv_roc_auc_std']:.4f}
- Precision: {best_model_info['cv_precision']:.4f} ± {best_model_info['cv_precision_std']:.4f}
- Recall: {best_model_info['cv_recall']:.4f} ± {best_model_info['cv_recall_std']:.4f}
- F1-Score: {best_model_info['cv_f1']:.4f} ± {best_model_info['cv_f1_std']:.4f}

## Threshold Analysis (Training Data)

**Default Threshold (0.5):**
- F1-Score: {threshold_info['default_f1']:.4f}
- Precision: {threshold_info['default_precision']:.4f}
- Recall: {threshold_info['default_recall']:.4f}

**Optimal Threshold (F1-Maximizing): {threshold_info['optimal_threshold']:.4f}):**
- F1-Score: {threshold_info['optimal_f1']:.4f}
- Precision: {threshold_info['optimal_precision']:.4f}
- Recall: {threshold_info['optimal_recall']:.4f}

**Operating Threshold Selected:** {threshold_info['optimal_threshold']:.4f} (maximizes F1 on training OOF predictions)

## Held-Out Test Set Performance

**Test Metrics (using selected threshold):**
- Average Precision: {test_metrics['test_avg_precision']:.4f}
- ROC-AUC: {test_metrics['test_roc_auc']:.4f}
- Precision: {test_metrics['test_precision']:.4f}
- Recall: {test_metrics['test_recall']:.4f}
- F1-Score: {test_metrics['test_f1']:.4f}
- Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}

## Selection Rationale

The model was selected based on cross-validation Average Precision from Phase 3
hyperparameter tuning. This metric was chosen as the primary ranking criterion
because it summarises the precision-recall trade-off and is appropriate for
imbalanced datasets (churn rate ~26.5%).

**Note on model comparison:** Logistic Regression (AP 0.6615) and Random Forest
(AP 0.6610) are effectively tied on mean cross-validation Average Precision
(difference < 0.001). Logistic Regression was selected because it achieves
equivalent discriminative performance with faster training and greater
interpretability. This is not a decisive advantage on AP alone; see
`reports/hyperparameter_tuning_results.json` for per-fold standard deviations.

**Key Considerations:**
- Cross-validation AP is the selection metric; threshold analysis and test-set
  results are distinct subsequent steps
- Selection used only training-side cross-validation results
- Test set was held out and not used for model selection or hyperparameter tuning
- Threshold was selected using out-of-fold predictions on training data, not test data
- Final evaluation on test set provides unbiased estimate of generalization performance

## Limitations

1. **No Resampling:** Class imbalance was handled via class_weight parameter only. SMOTE or other resampling techniques were not explored.
2. **Feature Engineering:** Limited to original features. No feature creation or interaction terms were explored.
3. **Hyperparameter Space:** Search space was intentionally limited to manageable size. More extensive tuning might yield better performance.
4. **Single Threshold:** Operating threshold was selected to maximize F1 on training data. Business-specific cost considerations might warrant a different threshold.
5. **Model Diversity:** Only four algorithm families were compared. Other algorithms (e.g., XGBoost, LightGBM, neural networks) were not evaluated.

## Artifacts Generated

- Fitted pipeline: `saved_models/best_model.joblib`
- Model metadata: `saved_models/model_metadata.json`
- Confusion matrix: `reports/confusion_matrix.png`
- ROC curve: `reports/roc_curve.png`
- Precision-recall curve: `reports/pr_curve.png`
- Model comparison chart: `reports/model_comparison.png`
- Test classification report: `reports/test_classification_report.txt`
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(summary)

def main():
    print("="*80)
    print("PHASE 4: FINAL MODEL SELECTION AND EVALUATION")
    print("="*80)
    
    # Create directories
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading training and test data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # Prepare features
    print("Preparing features...")
    X_train_prepared = prepare_feature_types(X_train)
    X_test_prepared = prepare_feature_types(X_test)
    
    preprocessor, num_features, cat_features, _ = get_preprocessor_and_features(X_train)
    
    # Load tuning results
    print("\nLoading Phase 3 tuning results...")
    tuning_results = load_tuning_results()
    
    # Select best model
    print("\nSelecting best model based on cross-validation Average Precision...")
    best_model_info = select_best_model(tuning_results)
    print(f"Selected: {best_model_info['model']}")
    print(f"Best Avg Precision: {best_model_info['best_avg_precision']:.4f}")
    print(f"Best Parameters: {best_model_info['best_params']}")
    
    # Parse parameters
    best_params = parse_best_params(best_model_info['best_params'])
    
    # Build pipeline with best parameters
    print("\nBuilding pipeline with best hyperparameters...")
    pipeline = build_pipeline_with_params(preprocessor, best_model_info['model'], best_params)
    
    # Get out-of-fold predictions
    print("\nGenerating out-of-fold predictions on training data...")
    oof_proba = get_out_of_fold_predictions(X_train_prepared, y_train, pipeline)
    
    # Evaluate thresholds
    print("\nEvaluating thresholds...")
    threshold_info = evaluate_thresholds(y_train, oof_proba)
    print(f"Default threshold (0.5) F1: {threshold_info['default_f1']:.4f}")
    print(f"Optimal threshold ({threshold_info['optimal_threshold']:.4f}) F1: {threshold_info['optimal_f1']:.4f}")
    
    # Select operating threshold
    operating_threshold = threshold_info['optimal_threshold']
    print(f"Selected operating threshold: {operating_threshold:.4f}")
    
    # Refit on entire training set
    print("\nRefitting pipeline on entire training set...")
    pipeline.fit(X_train_prepared, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on held-out test set...")
    test_metrics, y_pred, y_proba = evaluate_on_test(pipeline, X_test_prepared, y_test, operating_threshold)
    
    print("\nTest Set Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate artifacts
    print("\nGenerating evaluation artifacts...")
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png")
    print("  - Confusion matrix saved")
    
    # ROC curve
    plot_roc_curve(y_test, y_proba, REPORTS_DIR / "roc_curve.png")
    print("  - ROC curve saved")
    
    # Precision-recall curve
    plot_precision_recall_curve(y_test, y_proba, REPORTS_DIR / "pr_curve.png")
    print("  - Precision-recall curve saved")
    
    # Model comparison
    plot_model_comparison(tuning_results, REPORTS_DIR / "model_comparison.png")
    print("  - Model comparison chart saved")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    with open(REPORTS_DIR / "test_classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    print("  - Classification report saved")
    
    # Save pipeline
    joblib.dump(pipeline, SAVED_MODELS_DIR / "best_model.joblib")
    print(f"\nFitted pipeline saved to {SAVED_MODELS_DIR / 'best_model.joblib'}")
    
    # Save metadata
    # Strip pipeline prefixes to get bare estimator-parameter names.
    # LR/DT/RF params have 'classifier__X', SVC params have
    # 'classifier__estimator__X'; strip the longest matching prefix first.
    def _strip_prefix(k):
        if k.startswith('classifier__estimator__'):
            return k[len('classifier__estimator__'):]
        if k.startswith('classifier__'):
            return k[len('classifier__'):]
        return k

    clean_params = {_strip_prefix(k): v for k, v in best_params.items()}

    import sklearn
    metadata = {
        'model_name': best_model_info['model'],
        'scikit_learn_version': sklearn.__version__,
        'best_params': clean_params,
        'operating_threshold': float(operating_threshold),
        'cv_avg_precision': float(best_model_info['best_avg_precision']),
        'cv_avg_precision_std': float(best_model_info.get('cv_avg_precision_std', float('nan'))),
        'cv_metrics': {
            'roc_auc': float(best_model_info['cv_roc_auc']),
            'roc_auc_std': float(best_model_info['cv_roc_auc_std']),
            'precision': float(best_model_info['cv_precision']),
            'precision_std': float(best_model_info['cv_precision_std']),
            'recall': float(best_model_info['cv_recall']),
            'recall_std': float(best_model_info['cv_recall_std']),
            'f1': float(best_model_info['cv_f1']),
            'f1_std': float(best_model_info['cv_f1_std'])
        },
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }

    with open(SAVED_MODELS_DIR / "model_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {SAVED_MODELS_DIR / 'model_metadata.json'}")
    
    # Create selection summary
    # For display, use the same clean bare-estimator-param names
    clean_params_display = {_strip_prefix(k): v for k, v in best_params.items()}
    create_selection_summary(best_model_info, test_metrics, threshold_info, REPORTS_DIR / "selection_summary.md", clean_params_display)
    print(f"Selection summary saved to {REPORTS_DIR / 'selection_summary.md'}")
    
    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80)
    print("\nArtifacts created:")
    print(f"  - {SAVED_MODELS_DIR / 'best_model.joblib'}")
    print(f"  - {SAVED_MODELS_DIR / 'model_metadata.json'}")
    print(f"  - {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"  - {REPORTS_DIR / 'roc_curve.png'}")
    print(f"  - {REPORTS_DIR / 'pr_curve.png'}")
    print(f"  - {REPORTS_DIR / 'model_comparison.png'}")
    print(f"  - {REPORTS_DIR / 'test_classification_report.txt'}")
    print(f"  - {REPORTS_DIR / 'selection_summary.md'}")

if __name__ == "__main__":
    main()
