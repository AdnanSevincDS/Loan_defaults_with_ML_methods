from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier


ClassifierModel = Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier,XGBClassifier]

def make_prediction(model: ClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }