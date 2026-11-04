"""Ensemble-size, bias-variance, and training-label-noise experiments."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    ENSEMBLE_SIZES,
    NOISE_RATES,
    NOISE_SEEDS,
    SEED,
    TABLES,
)
from src.evaluate import fitted_model_map
from src.models import model_factories


def ensemble_size_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, float]:
    """Plot-ready five-fold accuracy curves using only training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    factories = {
        "Bagging": lambda n: BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            n_estimators=n,
            max_samples=best_params["Bagging"]["max_samples"],
            random_state=SEED,
        ),
        "Random forest": lambda n: RandomForestClassifier(
            n_estimators=n,
            max_features=best_params["Random forest"]["max_features"],
            random_state=SEED,
        ),
        "AdaBoost": lambda n: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            n_estimators=n,
            learning_rate=best_params["AdaBoost"]["learning_rate"],
            random_state=SEED,
        ),
        "Gradient boosting": lambda n: GradientBoostingClassifier(
            n_estimators=n,
            learning_rate=best_params["Gradient boosting"]["learning_rate"],
            max_depth=best_params["Gradient boosting"]["max_depth"],
            random_state=SEED,
        ),
    }
    rows = []
    for model_name, factory in factories.items():
        for count in ENSEMBLE_SIZES:
            estimator = factory(count)
            scores = cross_val_score(
                estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
            )
            rows.append(
                {
                    "model": model_name,
                    "n_estimators": count,
                    "mean_accuracy": float(np.mean(scores)),
                    "std_accuracy": float(np.std(scores, ddof=1)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "ensemble_size.csv", index=False)
    tree_scores = cross_val_score(
        DecisionTreeClassifier(random_state=SEED),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    return result, float(np.mean(tree_scores))


def bias_variance_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, dict[str, object]],
    cv_accuracy: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare training accuracy and repeated-CV accuracy for four learners."""
    models = fitted_model_map(best_params)
    names = ["Decision tree", "Decision stump", "Bagging", "AdaBoost"]
    rows = []
    for name in names:
        model = models[name]
        model.fit(X_train, y_train)
        rows.append(
            {
                "model": name,
                "training_accuracy": float(model.score(X_train, y_train)),
                "mean_cv_accuracy": float(np.mean(cv_accuracy[name])),
                "std_cv_accuracy": float(np.std(cv_accuracy[name], ddof=1)),
                "train_minus_cv_gap": float(
                    model.score(X_train, y_train) - np.mean(cv_accuracy[name])
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "bias_variance.csv", index=False)
    return result


def noise_robustness_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Flip labels in a training-only fit split and score a clean validation split."""
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=SEED,
    )
    factories = model_factories()
    rows = []
    for model_name, factory in factories.items():
        model_scores: dict[float, list[float]] = {rate: [] for rate in NOISE_RATES}
        for rate in NOISE_RATES:
            for noise_seed in NOISE_SEEDS:
                noisy_y = y_fit.copy()
                rng = np.random.default_rng(noise_seed)
                flip_count = int(round(rate * len(noisy_y)))
                if flip_count:
                    flipped = rng.choice(len(noisy_y), size=flip_count, replace=False)
                    noisy_y.iloc[flipped] = 1 - noisy_y.iloc[flipped].to_numpy()
                model = factory()
                if model_name in best_params:
                    model.set_params(**best_params[model_name])
                # A common ensemble size isolates noise response while keeping
                # this repeated experiment practical to run on a laptop.
                if model_name in {"Bagging", "Random forest", "AdaBoost", "Gradient boosting"}:
                    model.set_params(n_estimators=50)
                model.fit(X_fit, noisy_y)
                model_scores[rate].append(float(model.score(X_valid, y_valid)))
        for rate, values in model_scores.items():
            rows.append(
                {
                    "model": model_name,
                    "noise_rate": rate,
                    "mean_accuracy": float(np.mean(values)),
                    "std_accuracy": float(np.std(values, ddof=1)),
                    "n_runs": len(values),
                }
            )
        print(f"Completed label-noise trials for {model_name}")
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "noise_robustness.csv", index=False)
    return result
