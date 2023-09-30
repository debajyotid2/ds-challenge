"""
Functions to load and preprocess data.
"""
from typing import Literal
from pathlib import Path
import pandas as pd


def get_non_cat_cols(data: pd.DataFrame) -> set[str]:
    """Returns a set of columns from data that are non-categorical features."""
    cat_cols = set()
    for count, val in enumerate(data.dtypes):
        if val not in ["object", "str"]:
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


def extract_features(preprocessed_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Performs feature engineering on the cleaned dataset to prepare
    it for training a machine learning model.
    """
    features = preprocessed_data.copy(deep=True)
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

    return features
