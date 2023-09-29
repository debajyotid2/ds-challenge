"""
Functions to load and preprocess data.
"""
from pathlib import Path
import pandas as pd


def load_data(data_path: Path | str) -> pd.DataFrame:
    """Loads data from a text/csv/tsv/excel file."""
    data_path = Path(data_path)
    return pd.read_csv(data_path, delimiter=" ", header=None)
