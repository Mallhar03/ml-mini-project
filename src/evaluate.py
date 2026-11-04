"""Cross-validation, corrected comparisons, and final hold-out evaluation."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import make_scorer

from src.config import CV_REPEATS, CV_SPLITS, RESULTS, SEED, TABLES
from src.models import ensemble_grids, model_factories

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def tune_ensembles(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Tune each ensemble using five-fold CV on training data only."""
    factories = model_factories()
    best_params: dict[str, dict[str, object]] = {}
    best_estimators: dict[str, object] = {}
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    for name, grid in ensemble_grids().items():
        search = GridSearchCV(
            factories[name](),
            grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params[name] = search.best_params_
        best_estimators[name] = search.best_estimator_
        print(f"Tuned {name}: {search.best_params_} (CV accuracy={search.best_score_:.4f})")

    pd.DataFrame(
        [
            {"model": name, **params}
            for name, params in best_params.items()
        ]
    ).to_csv(TABLES / "best_params.csv", index=False)
    return best_params, best_estimators


def fitted_model_map(best_params: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Build fresh models and apply the selected ensemble parameters."""
    models = {}
    for name, factory in model_factories().items():
        model = factory()
        if name in best_params:
            model.set_params(**best_params[name])
        models[name] = model
    return models


def repeated_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Measure all models on the same 50 stratified training folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED
    )
    rows = []
    accuracy_scores = {}
    for name, model in models.items():
        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1,
        )
        accuracy_scores[name] = result["test_accuracy"]
        row = {"model": name}
        for metric in SCORING:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        row["mean_fit_time_seconds"] = float(np.mean(result["fit_time"]))
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        "accuracy_mean", ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(TABLES / "cv_comparison.csv", index=False)
    return comparison, accuracy_scores


def corrected_resampled_t_test(
    first: np.ndarray, second: np.ndarray, test_train_ratio: float = 0.25
) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio, 2003).

    Corrects the variance of paired cross-validation differences for the
    overlap between training sets: SE = sqrt((1/n + n_test/n_train) * s²).
    """
    differences = np.asarray(first) - np.asarray(second)
    n = len(differences)
    if n < 2:
        raise ValueError("At least two paired cross-validation scores are required.")
    standard_error = np.sqrt(
        (1 / n + test_train_ratio) * np.var(differences, ddof=1)
    )
    if standard_error == 0:
        return (0.0, 1.0) if np.mean(differences) == 0 else (
            float(np.sign(np.mean(differences)) * np.inf),
            0.0,
        )
    statistic = float(np.mean(differences) / standard_error)
    p_value = float(2 * student_t.sf(abs(statistic), df=n - 1))
    return statistic, p_value


def compare_significance(
    scores: Mapping[str, np.ndarray], comparison: pd.DataFrame
) -> pd.DataFrame:
    """Compare the best family members and best ensemble with the tree baseline."""
    bagging_models = ["Bagging", "Random forest"]
    boosting_models = ["AdaBoost", "Gradient boosting"]
    best_bagging = max(bagging_models, key=lambda name: comparison.set_index("model").loc[name, "accuracy_mean"])
    best_boosting = max(boosting_models, key=lambda name: comparison.set_index("model").loc[name, "accuracy_mean"])
    pairs = [
        (f"{best_bagging} vs {best_boosting}", best_bagging, best_boosting),
        ("Best ensemble vs Decision tree", max(
            [*bagging_models, *boosting_models],
            key=lambda name: comparison.set_index("model").loc[name, "accuracy_mean"],
        ), "Decision tree"),
    ]
    rows = []
    for label, first_name, second_name in pairs:
        first, second = scores[first_name], scores[second_name]
        statistic, corrected_p = corrected_resampled_t_test(first, second)
        try:
            wilcoxon_p = float(wilcoxon(first, second, alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "comparison": label,
                "model_1": first_name,
                "model_2": second_name,
                "mean_accuracy_difference": float(np.mean(first - second)),
                "nadeau_bengio_t": statistic,
                "nadeau_bengio_p_value": corrected_p,
                "wilcoxon_p_value_secondary": wilcoxon_p,
                "significant_at_0.05": corrected_p < 0.05,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "significance_tests.csv", index=False)
    return result


def final_holdout_evaluation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit on all training data and evaluate each model once on the test set."""
    rows = []
    predictions = {}
    probabilities = {}
    report_lines = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        positive_scores = model.predict_proba(X_test)[:, 1]
        predictions[name] = predicted
        probabilities[name] = positive_scores
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision_malignant": precision_score(y_test, predicted, zero_division=0),
                "recall_malignant": recall_score(y_test, predicted, zero_division=0),
                "f1_malignant": f1_score(y_test, predicted, zero_division=0),
                "roc_auc": roc_auc_score(y_test, positive_scores),
            }
        )
        report_lines.append(f"=== {name} ===")
        report_lines.append(
            classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                zero_division=0,
            )
        )
    result = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    result.to_csv(TABLES / "test_metrics.csv", index=False)
    (TABLES / "classification_reports.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return result, predictions, probabilities
