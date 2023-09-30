"""
Main script for the data science challenge
"""
import argparse
from pathlib import Path
from src.data import load_data, preprocess_data, extract_features


def main():
    parser = argparse.ArgumentParser(
        description="Main script for the data science challenge."
    )
    parser.add_argument("-p", "--data_path", help="Path to csv/text/excel data file.")
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        help="Window size to consider for rolling statistics for features. Default: 10",
        default=10,
    )
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError("Must supply a data path.")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError("Must supply a valid data path.")

    raw = load_data(args.data_path)
    preprocessed = preprocess_data(raw)
    breakpoint()
    features = extract_features(preprocessed, args.window_size)
    breakpoint()


if __name__ == "__main__":
    main()
