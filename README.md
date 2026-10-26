# Consumer Churn Prediction

A portfolio-grade machine learning project for predicting customer churn using Python and scikit-learn.

## Business Problem

Customer churn is a critical challenge for telecommunications companies. Identifying customers at risk of leaving enables proactive retention efforts, reducing revenue loss and improving customer lifetime value. This project builds a predictive model to classify customers as likely to churn or not, based on their service usage, account details, and demographic information.

## Project Objective

Develop a robust, production-ready machine learning pipeline that:
- Predicts customer churn with high precision and recall
- Handles class imbalance appropriately
- Prevents data leakage during preprocessing
- Provides interpretable results and clear decision rationale
- Is reproducible and thoroughly tested

## Dataset

**Source:** IBM Watson Telco Customer Churn Dataset (publicly available for educational purposes)

**Attribution:** This dataset is derived from IBM Sample Data Sets and is commonly used in machine learning education and research.

**Licensing:** The dataset is publicly available for educational and research purposes. Users should verify licensing terms before using for commercial applications. The dataset is commonly available on platforms like Kaggle and IBM's sample data repository.

**Note:** This repository includes the processed train/test splits (`data/processed/`) for reproducibility. The raw dataset (`data/raw/customer_churn.csv`) should be obtained from the original source or Kaggle.

**Dataset Size:** 7,043 customers with 21 features

**Target Variable:** Churn (Yes/No) - 26.5% churn rate

**Key Features:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account info: tenure, Contract, PaperlessBilling, PaymentMethod
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Charges: MonthlyCharges, TotalCharges

## Repository Architecture

```
consumer_churn_prediction/
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
├── algorithms/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── support_vector_classifier.py
│   └── model_registry.py
├── tests/
│   ├── test_data_pipeline.py
│   └── test_model_pipeline.py
├── reports/
│   ├── baseline_comparison.csv
│   ├── hyperparameter_tuning_summary.csv
│   ├── hyperparameter_tuning_results.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── model_comparison.png
│   ├── test_classification_report.txt
│   └── selection_summary.md
├── saved_models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── eda_and_outliers.ipynb
├── baseline_comparison.py
├── hyperparameter_tuning.py
├── final_evaluation.py
├── requirements.txt
└── pytest.ini
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Features and Preprocessing

### Feature Engineering Decisions

**Numerical Features (3):**
- `tenure`: Number of months customer has been with company
- `MonthlyCharges`: Monthly bill amount
- `TotalCharges`: Total charges over customer lifetime (converted from string to numeric)

**Categorical Features (16):**
- All other features except numerical and ID
- `SeniorCitizen` explicitly treated as categorical (converted from 0/1 integer to string)

**Preprocessing Pipeline:**
- Numerical: SimpleImputer (constant=0) → StandardScaler
- Categorical: SimpleImputer (most_frequent) → OneHotEncoder (handle_unknown='ignore')
- Missing values in `TotalCharges` (11 cases, all tenure=0) imputed with 0
- Rationale: New customers with tenure=0 have no accumulated charges

**Data Split:**
- Stratified 80:20 train-test split
- Random state: 42
- Preserves class distribution (26.5% churn)

## Candidate Algorithms

### 1. Logistic Regression
- Linear classifier for binary classification
- Outputs calibrated probabilities
- Interpretable coefficients
- Baseline: C=1, class_weight='balanced', max_iter=1000
- Tuned: C=1, class_weight=None (selected)

### 2. Decision Tree
- Non-linear classifier that learns decision rules
- Can capture complex interactions
- Prone to overfitting without regularization
- Baseline: max_depth=None, class_weight='balanced'
- Tuned: max_depth=5, min_samples_leaf=10, class_weight=None (selected)

### 3. Random Forest
- Ensemble of decision trees
- Reduces overfitting through bagging
- Handles non-linear relationships well
- Baseline: n_estimators=100, class_weight='balanced'
- Tuned: n_estimators=200, max_depth=10, max_features='sqrt', min_samples_leaf=5, class_weight=None (selected)

### 4. Support Vector Classifier
- Finds optimal hyperplane for classification
- Effective in high-dimensional spaces
- Requires probability calibration for threshold tuning
- Baseline: kernel='rbf', class_weight='balanced', probability=True
- Tuned: kernel='rbf', C=0.1, gamma='auto', class_weight='balanced (selected)

## Methodology

### Cross-Validation Strategy
- Stratified 5-fold cross-validation
- Shuffle=True, random_state=42
- Same fold assignments for all candidates
- Preprocessor fitted within each fold (no leakage)

### Selection Metric
**Primary:** Average Precision (AP)
- Summarizes precision-recall trade-off
- Appropriate for imbalanced datasets
- Focuses on positive class (churn) performance

**Supporting Metrics:**
- ROC-AUC: Overall discriminative ability
- Precision: Accuracy of positive predictions
- Recall: Coverage of actual churners
- F1-Score: Harmonic mean of precision and recall
- Balanced Accuracy: Average of sensitivity and specificity

### Hyperparameter Tuning
- GridSearchCV with same CV configuration
- Search spaces limited to manageable size
- Average Precision as refit metric
- Multi-metric scoring for all supporting metrics

## Results

### Baseline Comparison (Default Hyperparameters)

| Model | Avg Precision | ROC-AUC | F1-Score | Precision | Recall |
|-------|--------------|---------|----------|-----------|--------|
| Logistic Regression | 0.6601 | 0.8460 | 0.6282 | 0.5169 | 0.8013 |
| Random Forest | 0.6007 | 0.8216 | 0.6025 | 0.5676 | 0.6421 |
| Support Vector Classifier | 0.6065 | 0.8267 | 0.6236 | 0.5197 | 0.7799 |
| Decision Tree | 0.3926 | 0.6665 | 0.5094 | 0.5155 | 0.5037 |

### Tuned Model Comparison

| Model | Avg Precision | ROC-AUC | F1-Score | Precision | Recall |
|-------|--------------|---------|----------|-----------|--------|
| Logistic Regression | 0.6615 | 0.8461 | 0.5924 | 0.6521 | 0.5438 |
| Random Forest | 0.6610 | 0.8459 | 0.5794 | 0.6779 | 0.5064 |
| Support Vector Classifier | 0.6559 | 0.8403 | 0.6028 | 0.6459 | 0.5659 |
| Decision Tree | 0.6155 | 0.8288 | 0.5652 | 0.6325 | 0.5144 |

### Final Test Set Performance (Selected Model)

**Model:** Logistic Regression (C=1, class_weight=None)

**Operating Threshold:** 0.3273 (F1-maximizing on training OOF predictions)

| Metric | Value |
|--------|-------|
| Average Precision | 0.6337 |
| ROC-AUC | 0.8420 |
| Precision | 0.5378 |
| Recall | 0.7219 |
| F1-Score | 0.6164 |
| Balanced Accuracy | 0.7489 |

**Classification Report:**
```
              precision    recall  f1-score   support
    No Churn       0.89      0.78      0.83      1035
       Churn       0.54      0.72      0.62       374
    accuracy                           0.76      1409
```

### Evaluation Visualizations

- **Confusion Matrix:** Shows 270 true positives, 104 false positives, 270 false negatives, 765 true negatives
- **ROC Curve:** AUC of 0.8420, indicating good discriminative ability
- **Precision-Recall Curve:** AP of 0.6337, showing trade-off between precision and recall
- **Model Comparison Chart:** Visual comparison of all four tuned models across metrics

## Model Selection Decision

**Winner:** Logistic Regression

**Evidence:**
1. Highest cross-validation Average Precision (0.6615)
2. Highest ROC-AUC (0.8461)
3. Consistent performance across folds (low std)
4. Simple, interpretable model
5. Fast training and inference

**Trade-offs:**
- Random Forest achieved similar AP (0.6610) with higher precision but lower recall
- SVC achieved best F1-score (0.6028) but took 4x longer to train
- Logistic Regression provides best balance of performance, speed, and interpretability

## Threshold Trade-offs

**Default Threshold (0.5):**
- F1-Score: 0.5930
- Precision: 0.6520
- Recall: 0.5438
- Conservative: fewer false positives, misses more churners

**Optimal Threshold (0.3273):**
- F1-Score: 0.6381
- Precision: 0.5591
- Recall: 0.7431
- Aggressive: catches more churners, more false positives

**Business Considerations:**
- Lower threshold increases recall (catches 74% of churners vs 54%)
- Cost of false positive (retention offer to non-churner) vs cost of false negative (lost customer)
- Threshold can be adjusted based on business-specific cost structure

## Potential Business Use

**Deployment Scenarios:**
1. **Batch Scoring:** Daily/weekly scoring of customer base to identify at-risk customers
2. **Real-time Scoring:** API endpoint for individual customer churn prediction
3. **Retention Campaigns:** Targeted offers to high-risk customers
4. **Customer Service:** Alert agents when high-risk customers contact support

**Integration Points:**
- CRM system for customer data
- Marketing automation for retention campaigns
- Customer service dashboard for risk indicators

**Monitoring Requirements:**
- Track prediction drift over time
- Monitor actual churn rates vs predicted
- Retrain periodically with new data

## Installation and Reproduction

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
```bash
git clone <repository-url>
cd consumer_churn_prediction
pip install -r requirements.txt
```

### Reproduce Results

**Step 1: Data Preparation**
```bash
python src/data_loader.py
```
Generates `data/processed/train.csv` and `data/processed/test.csv`

**Step 2: Baseline Comparison**
```bash
python baseline_comparison.py
```
Output: `reports/baseline_comparison.csv`

**Step 3: Hyperparameter Tuning**
```bash
python hyperparameter_tuning.py
```
Output: `reports/hyperparameter_tuning_summary.csv`, `reports/hyperparameter_tuning_results.json`

**Step 4: Final Evaluation**
```bash
python final_evaluation.py
```
Output: Model artifacts and evaluation reports in `reports/` and `saved_models/`

**Step 5: Testing**
```bash
pytest -v
```
All 38 tests should pass

### Load Saved Model
```python
import joblib
import pandas as pd
from src.preprocessing import prepare_feature_types

# Load pipeline
pipeline = joblib.load('saved_models/best_model.joblib')

# Load metadata
import json
with open('saved_models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Make prediction
new_data = pd.DataFrame({...})  # Your data
new_data_prepared = prepare_feature_types(new_data)
prediction = pipeline.predict(new_data_prepared)
probability = pipeline.predict_proba(new_data_prepared)[:, 1]
```

## Limitations

1. **No Resampling Techniques:** Class imbalance handled via class_weight parameter only. SMOTE, ADASYN, or other resampling methods were not explored.

2. **Limited Feature Engineering:** Project uses original features only. No feature creation, interaction terms, or domain-specific transformations were explored.

3. **Hyperparameter Search Space:** Search space was intentionally limited to manageable size for computational efficiency. More extensive tuning might yield better performance.

4. **Algorithm Diversity:** Only four algorithm families were compared. Other algorithms (XGBoost, LightGBM, CatBoost, neural networks) were not evaluated.

5. **Threshold Selection:** Operating threshold selected to maximize F1 on training data. Business-specific cost considerations (cost of false positive vs false negative) were not incorporated.

6. **Temporal Validation:** No time-based validation was performed. Model assumes stationary distribution over time.

7. **Feature Importance:** Limited interpretability beyond Logistic Regression coefficients. SHAP or other explainability methods not implemented.

8. **Dataset Specificity:** Model trained on telecom-specific data. Performance may not generalize to other industries or datasets.

9. **No A/B Testing:** Model performance not validated in production environment with actual business outcomes.

10. **Single Dataset:** Model evaluated on single train-test split. No external validation on different datasets.

## Testing

The project includes 38 pytest tests covering:
- Data loading and validation (8 tests)
- Preprocessing pipeline (10 tests)
- Data leakage prevention (2 tests)
- Preprocessor compatibility with all classifiers (4 tests)
- Unknown categorical value handling (1 test)
- Model training and prediction (3 tests)
- Prediction probability validation (2 tests)
- Feature consistency (2 tests)
- Saved pipeline loading (2 tests)
- Reproducibility (2 tests)
- Target leakage prevention (2 tests)
- Integration test (1 test)

Run tests:
```bash
pytest -v
```

## Key Features

- **Data Leakage Prevention:** Preprocessor remains unfitted until cross-validation
- **Class Imbalance Handling:** Uses class_weight parameter (no SMOTE)
- **Reproducibility:** Fixed random_state=42 throughout
- **Comprehensive Testing:** 38 pytest tests covering data pipeline, model training, and inference
- **Threshold Optimization:** F1-maximizing threshold selected on training OOF predictions
- **Model Persistence:** Saved pipeline with metadata for inference
- **Unknown Category Handling:** OneHotEncoder configured to ignore unknown categories during inference
