from typing import Union

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.entities.train_params import LogRegParams,DecisionTreeParams, RandomForestParams, XGBoostParams

ClassifierModel = Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier,XGBClassifier]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: Union[LogRegParams,DecisionTreeParams, RandomForestParams, XGBoostParams],
                ) -> ClassifierModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            max_depth=train_params.max_depth,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=train_params.C,
            penalty=train_params.penalty,
            solver=train_params.solver,
            tol=train_params.tol,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(
            criterion=train_params.criterion,
            min_samples_split=train_params.min_samples_split,
            max_depth=train_params.max_depth,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "XGBClassifier":
        model = XGBClassifier(
            n_estimators=train_params.n_estimators,
            max_depth=train_params.max_depth,
            random_state=train_params.random_state,
            learning_rate=train_params.learning_rate,
            subsample=train_params.subsample,
        )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model