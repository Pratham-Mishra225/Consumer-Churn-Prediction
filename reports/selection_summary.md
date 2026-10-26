# Model Selection Summary

## Selected Model

**Model:** Logistic Regression

**Best Hyperparameters:**
```
{'C': 1, 'class_weight': None}
```

**Cross-Validation Performance (Training Data):**
- Average Precision: 0.6615
- ROC-AUC: 0.8461 ± 0.0126
- Precision: 0.6521 ± 0.0256
- Recall: 0.5438 ± 0.0409
- F1-Score: 0.5924 ± 0.0298

## Threshold Analysis (Training Data)

**Default Threshold (0.5):**
- F1-Score: 0.5930
- Precision: 0.6520
- Recall: 0.5438

**Optimal Threshold (F1-Maximizing): 0.3273):**
- F1-Score: 0.6381
- Precision: 0.5591
- Recall: 0.7431

**Operating Threshold Selected:** 0.3273 (maximizes F1 on training OOF predictions)

## Held-Out Test Set Performance

**Test Metrics (using selected threshold):**
- Average Precision: 0.6337
- ROC-AUC: 0.8420
- Precision: 0.5378
- Recall: 0.7219
- F1-Score: 0.6164
- Balanced Accuracy: 0.7489

## Selection Rationale

The model was selected based on cross-validation Average Precision from Phase 3 hyperparameter tuning. This metric was chosen as the primary ranking criterion because it summarizes precision-recall trade-off and is appropriate for imbalanced datasets (churn rate ~26.5%).

**Key Considerations:**
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
