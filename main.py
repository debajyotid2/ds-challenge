"""
Main script for the data science challenge
"""
import warnings

warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import (
    load_data,
    preprocess_data,
    extract_features,
    generate_labels,
    generate_train_val_test_idxs,
)
from src.model import (
    train_and_evaluate,
    predict_and_adjust,
    predict_and_adjust_against_gt,
    calculate_classification_metrics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Main script for the data science challenge."
    )
    parser.add_argument(
        "--training_data_path", help="Path to csv/text/excel training data file."
    )
    parser.add_argument(
        "--inference_data_path", help="Path to csv/text/excel inference data file."
    )
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

    if args.training_data_path is None:
        raise ValueError("Must supply a training data path.")

    training_data_path = Path(args.training_data_path)

    if args.inference_data_path is not None:
        inference_data_path = Path(args.inference_data_path)
        if not inference_data_path.exists():
            raise ValueError("Must supply a valid inference data path.")

    if not training_data_path.exists():
        raise ValueError("Must supply a valid training data path.")

    # Set random number generator seed
    np.random.seed(args.seed)

    # Load, preprocess and feature engineer data
    raw = load_data(training_data_path)
    preprocessed = preprocess_data(raw)
    machine_id_column = preprocessed[0]
    features = extract_features(preprocessed, args.feature_window_size)

    # Get labels
    labels = generate_labels(machine_id_column, args.failure_window_size)

    # Split data into training, validation and test
    train_idxs, val_idxs, test_idxs = generate_train_val_test_idxs(
        machine_id_column, args.val_frac, args.test_frac, args.seed
    )

    x_train, y_train = features.loc[train_idxs, :], labels.loc[train_idxs]
    x_val, y_val = features.loc[val_idxs, :], labels.loc[val_idxs]
    x_test, y_test = features.loc[test_idxs, :], labels.loc[test_idxs]

    # Train model
    models = dict.fromkeys(
        ["logistic", "svc", "decision tree", "random forest", "xgb", "mlp"],
        dict.fromkeys(["pipeline", "metrics"]),
    )
    best_roc_auc, best_model = 0.0, None
    for model in models:
        models[model]["pipeline"], models[model]["metrics"] = train_and_evaluate(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            machine_id_column[val_idxs],
            display_confusion_matrix=False,
        )
        if models[model]["metrics"]["roc_auc"] > best_roc_auc:
            best_roc_auc = models[model]["metrics"]["roc_auc"]
            best_model = models[model]["pipeline"]

    # Evaluate best model on test data
    y_pred = predict_and_adjust_against_gt(
        best_model, x_test, y_test, machine_id_column[test_idxs]
    )
    metrics = calculate_classification_metrics(y_pred, y_test)

    print("-----------------------------------------------------------")
    print(f"Best model based on ROC-AUC: {best_model['classifier']}:")
    print("Evaluation on test set:")
    print(f" Balanced accuracy score: {metrics['bal_acc']*100:.2f}%.")
    print(f" Precision score: {metrics['precision']*100:.2f}%.")
    print(f" Recall score: {metrics['recall']*100:.2f}%.")
    print(f" F1 score: {metrics['f1']*100:.2f}%.")
    print(f" ROC-AUC score: {metrics['roc_auc']*100:.2f}%.")
    print("-----------------------------------------------------------")

    # Perform inference on inference data
    if args.inference_data_path is None:
        return

    # Load, preprocess and feature engineer data
    raw = load_data(inference_data_path)
    preprocessed = preprocess_data(raw)
    machine_id_column = preprocessed[0]
    features = extract_features(preprocessed, args.feature_window_size)

    # Generate predictions
    y_pred = predict_and_adjust(best_model, features, machine_id_column)
    prediction_df = pd.concat([machine_id_column, preprocessed[1], y_pred], axis=1)
    prediction_df.columns = ["machine_ID", "cycle", "prediction"]

    # Write predictions to disk.
    inference_path = inference_data_path.with_name("predictions.csv")
    prediction_df.to_csv(inference_path, index=False)
    print(f"Predictions written to {inference_path.resolve()}.")

    # Display results of prediction
    print("-----------------------------------------------------------")
    print("Failing machines and the first predicted cycle of failure: ")
    machine_id_grp = prediction_df.groupby(by="machine_ID")
    aggregated = machine_id_grp.agg("sum")
    failing_machines = aggregated[aggregated["prediction"] > 0]
    for machine_id in failing_machines.index:
        machine_data = machine_id_grp.get_group(machine_id)
        filt = machine_data["prediction"] == 1.0
        first_pos_cycle = machine_data[filt].iloc[0, 1]
        print(
            f"Machine ID: {machine_id} will fail within next {int(args.failure_window_size)} cycles of cycle {int(first_pos_cycle)}."
        )
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    main()
