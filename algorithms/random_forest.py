from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100,
        max_depth=None,
        n_jobs=-1
    )
