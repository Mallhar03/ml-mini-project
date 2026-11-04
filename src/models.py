"""Model constructors and equally bounded ensemble tuning grids."""

from collections.abc import Callable

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED

ModelFactory = Callable[[], object]


def model_factories() -> dict[str, ModelFactory]:
    """Return constructors for all required models in report order."""
    return {
        "Most frequent": lambda: DummyClassifier(strategy="most_frequent"),
        "Decision tree": lambda: DecisionTreeClassifier(random_state=SEED),
        "Decision stump": lambda: DecisionTreeClassifier(
            max_depth=1, random_state=SEED
        ),
        "Bagging": lambda: BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            random_state=SEED,
        ),
        "Random forest": lambda: RandomForestClassifier(random_state=SEED),
        "AdaBoost": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            random_state=SEED,
        ),
        "Gradient boosting": lambda: GradientBoostingClassifier(random_state=SEED),
    }


def ensemble_grids() -> dict[str, dict[str, list[object]]]:
    """Return the requested small grid for each ensemble family member."""
    return {
        "Bagging": {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.5, 1.0],
        },
        "Random forest": {
            "n_estimators": [100, 200],
            "max_features": ["sqrt", "log2"],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],
        },
        "Gradient boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    }
