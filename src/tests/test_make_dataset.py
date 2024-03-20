from typing import NoReturn

import pandas as pd

from src.data.make_dataset import split_train_val_data
from src.entities.split_params import SplittingParams


def test_split_train_val_data(data:pd.DataFrame, target_col:str) -> NoReturn:
    """
    Test the function that splits the data into training and validation data
    """
    test_size = 0.3
    params = SplittingParams(val_size=test_size, random_state=42)
    X_train, X_test, y_train, y_test = split_train_val_data(data, target_col, params)

    tolerance = 0.0001

    assert len(X_train) >= len(data) * (0.7 - tolerance)
    assert len(X_test) <= len(data) * (0.3 + tolerance)
    assert len(X_train) + len(X_test) == len(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
