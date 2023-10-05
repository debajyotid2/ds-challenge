"""
Various helpers to load appropriate classification model.
"""

from typing import Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
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


def adjust_predictions_against_gt(
    y_pred: np.ndarray, y: pd.Series, machine_id_column: pd.Series
) -> pd.Series:
    """
    For every machine ID in set, takes the first true positive prediction
    and converts all predictions in the cycles following it to positive
    predictions.
    NOTE: This function works only if the true labels are available. Hence
    it cannot be used during inference.
    """
    y_pred = pd.Series(data=y_pred, index=y.index, name="predictions")
    for machine_id in machine_id_column.unique():
        machine_id_filt = machine_id_column == machine_id
        failure_window = y[machine_id_filt] == 1.0
        y_pred_flr_wdw = y_pred[machine_id_filt][failure_window]
        positive_preds_flr_wdw = y_pred_flr_wdw[y_pred_flr_wdw == 1.0]
        if len(positive_preds_flr_wdw) == 0:
            continue
        first_pos_idx = positive_preds_flr_wdw.index[0]
        last_pos_idx = y_pred_flr_wdw.index[-1]
        y_pred.loc[first_pos_idx:last_pos_idx] = 1.0
    return y_pred


def predict_and_adjust_against_gt(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    machine_id_column: pd.Series,
) -> pd.Series:
    """
    Adjust predictions to define all predictions within the failure prediction
    window (cycles) after the first positive prediction to be positive for each
    machine ID. This adjustment reflects real world expectations, as once a
    model predicts that a failure will occur within failure_window cycles, any
    negative predictions after that are insignificant and are treated as
    positive.  NOTE: This adjustment works only if ground truth labels are
    provided.
    """
    y_pred = pipeline.predict(x)
    y_pred = adjust_predictions_against_gt(y_pred, y, machine_id_column)
    return y_pred


def predict_and_adjust(
    pipeline: Pipeline,
    x: pd.DataFrame,
    machine_id_column: pd.Series,
) -> pd.Series:
    """
    Adjust predictions to define all predictions after the first positive
    prediction to be positive for each machine ID.  This adjustment reflects
    real world expectations, as once a model predicts that a failure will occur
    within failure_window cycles, any negative predictions after that are
    insignificant and are treated as positive.  NOTE: This function can be used
    during inference.
    """

    def adjust_predictions(
        y_pred: np.ndarray, machine_id_column: pd.Series
    ) -> pd.Series:
        """
        For every machine ID in set, takes the first positive prediction
        and converts all predictions in the cycles following it to positive
        predictions.
        """
        y_pred = pd.Series(
            data=y_pred, index=machine_id_column.index, name="predictions"
        )
        for machine_id in machine_id_column.unique():
            machine_id_filt = machine_id_column == machine_id
            y_pred_machine = y_pred[machine_id_filt]
            pos_pred_mask = y_pred_machine == 1.0
            if pos_pred_mask.sum() == 0.0:
                continue
            first_pos_idx = y_pred_machine[pos_pred_mask].index[0]
            last_pos_idx = y_pred_machine.index[-1]
            y_pred.loc[first_pos_idx:last_pos_idx] = 1.0
        return y_pred

    y_pred = pipeline.predict(x)
    return adjust_predictions(y_pred, machine_id_column)


def calculate_classification_metrics(
    y: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """Calculates classification metrics."""
    metrics = {}
    metrics["bal_acc"] = balanced_accuracy_score(y, y_pred)
    metrics["precision"] = precision_score(y, y_pred)
    metrics["recall"] = recall_score(y, y_pred)
    metrics["f1"] = f1_score(y, y_pred)
    metrics["roc_auc"] = roc_auc_score(y, y_pred)
    return metrics


def train_and_evaluate(
    model: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    machine_id_column_val: pd.Series,
    display_confusion_matrix: bool = True,
) -> tuple[Pipeline, dict[str, float]]:
    """Train and evaluate a model."""
    classifier = load_model(model)
    pipeline_list = []
    if model not in ["decision tree", "random forest", "xgb"]:
        pipeline_list.append(("scaler", RobustScaler()))
    pipeline_list.append(("classifier", classifier()))

    pipeline = Pipeline(pipeline_list)
    pipeline.fit(x_train, y_train)

    y_pred = predict_and_adjust_against_gt(
        pipeline, x_val, y_val, machine_id_column_val
    )

    metrics = calculate_classification_metrics(y_val, y_pred)

    print(f"Model {model}:")
    print(f" Balanced accuracy score: {metrics['bal_acc']*100:.2f}%.")
    print(f" Precision score: {metrics['precision']*100:.2f}%.")
    print(f" Recall score: {metrics['recall']*100:.2f}%.")
    print(f" F1 score: {metrics['f1']*100:.2f}%.")
    print(f" ROC-AUC score: {metrics['roc_auc']*100:.2f}%.")

    # Confusion matrix
    if display_confusion_matrix:
        ConfusionMatrixDisplay.from_estimator(pipeline, x_val, y_val)
        plt.show()

    return pipeline, metrics
