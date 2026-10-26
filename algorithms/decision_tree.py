from sklearn.tree import DecisionTreeClassifier

def get_model():
    """Return Decision Tree classifier with baseline settings."""
    return DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        max_depth=None,
        min_samples_split=2
    )
