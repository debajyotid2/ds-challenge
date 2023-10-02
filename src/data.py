"""
Functions to load and preprocess data.
"""
from typing import Literal
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd


def get_non_cat_cols(data: pd.DataFrame) -> set[str]:
    """
    Returns a set of columns from data that are non-categorical features.
    Columns with datatype object, string or int64 are considered categorical.
    """
    cat_cols = set()
    for count, val in enumerate(data.dtypes):
        if val not in ["object", "str", "int64"]:
            continue
        cat_cols.add(data.columns[count])
    return set(data.columns) - cat_cols


def load_data(data_path: Path | str) -> pd.DataFrame:
    """Loads data from a text/csv/tsv/excel file."""
    data_path = Path(data_path)
    return pd.read_csv(data_path, delimiter=" ", header=None)


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning and missing value imputation."""
    processed = raw_data.copy(deep=True)
    # Drop columns with null values
    processed = processed.dropna(axis=1, how="any")

    # Convert column representing time cycles to float
    processed[1] = processed[1].astype(np.float64)

    # Convert all sensor signal columns to float
    processed.iloc[:, 3:] = processed.iloc[:, 3:].astype(np.float64)

    # Remove non-categorical features that have zero variation, i.e. do
    # not change over time
    non_cat_cols = get_non_cat_cols(processed)
    unchanging_cols = [col for col in non_cat_cols if processed[col].std() <= 0.0001]
    processed.drop(columns=unchanging_cols, inplace=True)

    processed.sort_values(by=[0, 1], ascending=[True, True])

    return processed


def get_rolling_statistics(
    feature: pd.Series,
    statistic: Literal["mean", "median", "max", "min"],
    window_size: int = 10,
) -> pd.Series:
    """
    Gets rolling statistics within the window size specified. Skips the
    first window_size entries and keeps original values instead.
    """
    rolling_stat_feature = feature.copy(deep=True)
    rolling_stat = feature.rolling(window=window_size)
    match statistic:
        case "mean":
            rolling_stat_feature[window_size - 1 :] = rolling_stat.mean()[
                window_size - 1 :
            ]
        case "median":
            rolling_stat_feature[window_size - 1 :] = rolling_stat.median()[
                window_size - 1 :
            ]
        case "max":
            rolling_stat_feature[window_size - 1 :] = rolling_stat.max()[
                window_size - 1 :
            ]
        case "min":
            rolling_stat_feature[window_size - 1 :] = rolling_stat.max()[
                window_size - 1 :
            ]
        case other:
            raise ValueError(f"Invalid statistic {statistic}.")
    rolling_stat_feature.name = f"{feature.name}_moving_{statistic}"
    return rolling_stat_feature


def get_relative_change_signal(
    feature: pd.Series, feature_moving_mean: pd.Series, window_size: int
) -> pd.Series:
    """
    Creates a new feature containing the signal to mean ratio to emphasize
    sudden spikes or valleys in the signal.
    """
    signal_to_mean = feature.copy(deep=True)
    signal_to_mean[: window_size - 1] = 0.0
    feature_moving_mean = feature_moving_mean.copy(deep=True).replace(0.0, 1.0)
    signal_to_mean[window_size - 1 :] = (feature / feature_moving_mean)[
        window_size - 1 :
    ]
    signal_to_mean.fillna(0.0, inplace=True)
    signal_to_mean.name = f"{feature.name}_relative_change"
    return signal_to_mean


def extract_features(preprocessed_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Performs feature engineering on the cleaned dataset to prepare
    it for training a machine learning model.
    """
    features = preprocessed_data.copy(deep=True)
    features = features.sort_values(by=[0, 1], ascending=True).reset_index(drop=True)
    features.drop(columns=[0, 1], inplace=True)
    non_cat_cols = get_non_cat_cols(features)

    # Get rolling statistics
    moving_stats = []
    stats = ["mean", "median", "max", "min"]
    for col in non_cat_cols:
        for stat in stats:
            moving_stats.append(
                get_rolling_statistics(features[col], stat, window_size)
            )
    features = pd.concat([features, *moving_stats], axis=1)

    # Get the signal to mean ratio to emphasize sudden spikes or valleys
    # in signal.
    rel_change_stats = []
    for col in non_cat_cols:
        mean_col = f"{col}_moving_mean"
        rel_change_stats.append(
            get_relative_change_signal(features[col], features[mean_col], window_size)
        )
    features = pd.concat([features, *rel_change_stats], axis=1)

    features.columns = [str(col) for col in features.columns]

    return features


def generate_labels(
    machine_id_column: pd.Series, failure_window_size: int = 20
) -> pd.Series:
    """
    Based on the failure window size, generate target column indicating 1
    when a failure will occur within the next failure_window_size days and 0
    if not. It is assumed that the machine_id_column is sorted by machine ID
    and time cycle.
    """
    machine_ids = machine_id_column.unique()
    label_col = pd.Series(
        index=machine_id_column.index,
        data=np.zeros_like(machine_id_column.values, dtype=np.float32),
        name="failure",
    )
    for machine_id in machine_id_column:
        idxs = label_col[machine_id_column == machine_id][-failure_window_size:].index
        label_col[idxs] = 1.0
    return label_col


def generate_train_val_test_idxs(
    machine_id_column: pd.Series,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Given a set of machine IDs, sets aside val_frac fraction of machines for
    the validation set, test_frac fraction for the test set and the remaining
    for the training set. Returns the respective indexes from the machine id
    column.
    """

    def gather_subset_idxs(machine_ids: np.ndarray) -> pd.Index:
        filt = [(machine_id_column == id_val) for id_val in machine_ids]
        filt = reduce(lambda x, y: x | y, filt)
        return machine_id_column[filt].index

    machine_ids = machine_id_column.unique()
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(machine_ids)
    num_val, num_test = int(val_frac * len(machine_ids)), int(
        test_frac * len(machine_ids)
    )
    val_machine_ids = machine_ids[:num_val]
    test_machine_ids = machine_ids[num_val : num_val + num_test]
    train_machine_ids = machine_ids[num_val + num_test :]

    # Gather indices for each subset
    train_idxs = gather_subset_idxs(train_machine_ids)
    val_idxs = gather_subset_idxs(val_machine_ids)
    test_idxs = gather_subset_idxs(test_machine_ids)

    # Shuffle the training indices before returning
    rng.shuffle(train_idxs.to_numpy())

    return train_idxs, val_idxs, test_idxs
