import pandas as pd
from pathlib import Path


def func():
    pass


if __name__ == "__main.py__":
    data = pd.read_csv(Path('../data/duplicate_data_case_study.csv'))
    print(data.head())