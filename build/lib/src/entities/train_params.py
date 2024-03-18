from dataclasses import dataclass, field


@dataclass()
class LogRegParams:
    model_type: str = field(default="LogisticRegression")
    C: float = field(default=1.0)
    penalty: str = field(default="l2")
    solver: str = field(default="lbfgs")
    tol: float = field(default=1e-4)
    random_state: int = field(default=42)


@dataclass()
class DecisionTreeParams:
    model_type: str = field(default="DecisionTreeClassifier")
    criterion: str = field(default="gini")
    min_samples_split: int = field(default=2)
    max_depth: int = field(default=None)
    random_state: int = field(default=42)


@dataclass()
class RandomForestParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=100)
    max_depth: int = field(default=None)
    random_state: int = field(default=42)


@dataclass()
class XGBoostParams:
    model_type: str = field(default="XGBClassifier")
    n_estimators: int = field(default=100)
    max_depth: int = field(default=3)
    learning_rate: float = field(default=0.1)
    subsample: float = field(default=1)
    random_state: int = field(default=42)

