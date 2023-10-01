"""
Various helpers to load appropriate classification model.
"""

from typing import Any, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

MODEL_DICT = {
    "random forest": RandomForestClassifier,
    "decision tree": DecisionTreeClassifier,
    "svc": SVC,
    "mlp": MLPClassifier,
    "xgb": XGBClassifier,
    "logistic": LogisticRegression,
}


def load_model(model: str) -> Callable[Any, Any]:
    """
    Loads model from model dict given a model name.
    """
    if model not in MODEL_DICT:
        raise ValueError(f"Invalid model {model}. Must be one of {MODEL_DICT.keys()}.")
    return MODEL_DICT[model]
