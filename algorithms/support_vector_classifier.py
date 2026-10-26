from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def get_model():
    """Return Support Vector Classifier with baseline settings."""
    base_svc = SVC(
        random_state=42,
        class_weight='balanced',
        kernel='rbf'
    )
    return CalibratedClassifierCV(base_svc, ensemble=False)
