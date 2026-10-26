from algorithms.logistic_regression import get_model as get_logistic_regression
from algorithms.decision_tree import get_model as get_decision_tree
from algorithms.random_forest import get_model as get_random_forest
from algorithms.support_vector_classifier import get_model as get_svm

MODEL_REGISTRY = {
    "Logistic Regression": get_logistic_regression,
    "Decision Tree": get_decision_tree,
    "Random Forest": get_random_forest,
    "Support Vector Classifier": get_svm
}

def get_all_models():
    return MODEL_REGISTRY

def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    return MODEL_REGISTRY[model_name]()
