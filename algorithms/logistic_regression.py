from sklearn.linear_model import LogisticRegression

def get_model():
    """Return Logistic Regression classifier with baseline settings."""
    return LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
