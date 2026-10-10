
import pandas as pd
import pytest

from pybd_in.analysis import analyze_dataset
from pybd_in.preprocessing import recommend_preprocessing


def test_dataset_with_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 25],
        "salary": [30000, 40000, None],
    })

    result = analyze_dataset(df)

    assert result is not None


def test_dataset_with_duplicate_rows():
    df = pd.DataFrame({
        "name": ["Anik", "Rahul", "Anik"],
        "score": [80, 90, 80],
    })

    result = analyze_dataset(df)

    assert result is not None


def test_recommendations_return_result():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35],
        "salary": [20000, 30000, 40000, 50000],
    })

    result = recommend_preprocessing(df)

    assert result is not None
