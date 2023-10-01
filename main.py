"""
Main script for the data science challenge
"""
import warnings

warnings.filterwarnings("ignore")
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from src.data import (
    load_data,
    preprocess_data,
    extract_features,
    generate_labels,
    generate_train_val_test_idxs,
)
from src.model import load_model


def train_and_evaluate(
    model: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Train and evaluate a model."""
    classifier = load_model(model)
    pipeline_list = []
    if model not in ["decision tree", "random forest", "xgb"]:
        pipeline_list.append(("scaler", RobustScaler()))
    pipeline_list.append(("classifier", classifier()))

    pipeline = Pipeline(pipeline_list)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(f"Model {model}:")
    print(
        f" Balanced accuracy score: {balanced_accuracy_score(y_test,y_pred)*100:.2f}%."
    )
    print(f" Precision score: {precision_score(y_test,y_pred)*100:.2f}%.")
    print(f" Recall score: {recall_score(y_test,y_pred)*100:.2f}%.")
    print(f" F1 score: {f1_score(y_test,y_pred)*100:.2f}%.")
    ConfusionMatrixDisplay.from_estimator(pipeline, x_test, y_test)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Main script for the data science challenge."
    )
    parser.add_argument("-p", "--data_path", help="Path to csv/text/excel data file.")
    parser.add_argument(
        "--feature_window_size",
        type=int,
        help="Window size to consider for rolling statistics for features. Default: 10",
        default=10,
    )
    parser.add_argument(
        "--failure_window_size",
        type=int,
        help="Window size or number of cycles within which failure must be predicted. Default: 20",
        default=20,
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        help="Fraction of machines in the validation set. Default: 0.10",
        default=0.10,
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        help="Fraction of machines in the test set. Default: 0.10",
        default=0.10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random number seed. Default: 42",
        default=42,
    )
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError("Must supply a data path.")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError("Must supply a valid data path.")

    # Set random number generator seed
    np.random.seed(args.seed)

    # Load, preprocess and feature engineer data
    raw = load_data(args.data_path)
    preprocessed = preprocess_data(raw)
    features = extract_features(preprocessed, args.feature_window_size)
    features = features.sort_values(by=[0, 1], ascending=True).reset_index(drop=True)

    # Get labels
    labels = generate_labels(features[0], args.failure_window_size)

    # Split data into training, validation and test
    train_idxs, val_idxs, test_idxs = generate_train_val_test_idxs(
        features[0], args.val_frac, args.test_frac, args.seed
    )

    features.drop(columns=[0, 1], inplace=True)
    features.columns = [str(col) for col in features.columns]

    x_train, y_train = features.loc[train_idxs, :], labels.loc[train_idxs]
    x_val, y_val = features.loc[val_idxs, :], labels.loc[val_idxs]
    x_test, y_test = features.loc[test_idxs, :], labels.loc[test_idxs]

    # Train model
    models = ["logistic", "svc", "decision tree", "random forest", "xgb", "mlp"]

    for model in models:
        train_and_evaluate(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
