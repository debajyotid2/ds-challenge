"""
Main script for the data science challenge
"""
import argparse
from pathlib import Path
from src.data import load_data


def main():
    parser = argparse.ArgumentParser(
        description="Main script for the data science challenge."
    )
    parser.add_argument("-p", "--data_path", help="Path to csv/text/excel data file.")
    args = parser.parse_args()

    data = load_data(args.data_path)
    breakpoint()


if __name__ == "__main__":
    main()
