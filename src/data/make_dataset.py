from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities.split_params import SplittingParams


def split_train_val_data(
        data: pd.DataFrame,
        target_col: str,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.val_size, random_state=params.random_state, stratify=y, shuffle=True
    )
    return X_train, X_test, y_train, y_test