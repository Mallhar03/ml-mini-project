"""Report-oriented plots saved consistently at publication resolution."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import FIGURES

sns.set_theme(style="whitegrid")


def plot_eda(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Save class balance and mean-feature correlation plots."""
    labels = y_train.map({0: "Benign", 1: "Malignant"})
    counts = labels.value_counts().reindex(["Benign", "Malignant"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set(title="Training-set class distribution", xlabel="Diagnosis", ylabel="Samples")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_class_distribution.png", dpi=300)
    plt.close(fig)

    mean_features = [name for name in X_train.columns if name.startswith("mean ")]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        X_train[mean_features].corr(),
        cmap="vlag",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation of the 10 mean features (training set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_mean_feature_correlation.png", dpi=300)
    plt.close(fig)


def plot_cv_boxplot(scores: pd.DataFrame) -> None:
    """Plot all repeated-CV accuracy observations by model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=scores, x="model", y="accuracy", hue="model", legend=False, ax=ax)
    ax.set(title="Repeated stratified CV accuracy (training set)", xlabel="", ylabel="Accuracy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_cv_accuracy_boxplot.png", dpi=300)
    plt.close(fig)


def plot_holdout(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> None:
    """Save confusion matrices and ROC curves for final hold-out predictions."""
    model_names = list(predictions)
    columns = 3
    rows = (len(model_names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    for ax, name in zip(axes.flat, model_names):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            predictions[name],
            display_labels=["Benign", "Malignant"],
            ax=ax,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(name)
    for ax in axes.flat[len(model_names):]:
        ax.remove()
    fig.suptitle("Locked hold-out confusion matrices", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for name in model_names:
        RocCurveDisplay.from_predictions(
            y_true, probabilities[name], name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves on the locked hold-out set")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_roc_curves.png", dpi=300)
    plt.close(fig)


def plot_ensemble_size(
    results: pd.DataFrame, decision_tree_accuracy: float
) -> None:
    """Plot cross-validated accuracy against the number of estimators."""
    lower_accuracy = max(
        0.0,
        float((results["mean_accuracy"] - results["std_accuracy"]).min()) - 0.02,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in results.groupby("model", sort=False):
        group = group.sort_values("n_estimators")
        ax.plot(group["n_estimators"], group["mean_accuracy"], marker="o", label=model)
        ax.fill_between(
            group["n_estimators"],
            group["mean_accuracy"] - group["std_accuracy"],
            group["mean_accuracy"] + group["std_accuracy"],
            alpha=0.15,
        )
    ax.axhline(
        decision_tree_accuracy,
        linestyle="--",
        color="black",
        label="Decision tree (5-fold CV)",
    )
    ax.set(
        title="Ensemble size vs. training-set CV accuracy",
        xlabel="Number of estimators",
        ylabel="Mean 5-fold accuracy (shading: ±1 std)",
        ylim=(lower_accuracy, 1.01),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_ensemble_size.png", dpi=300)
    plt.close(fig)


def plot_noise_robustness(results: pd.DataFrame) -> None:
    """Plot validation accuracy across training-label noise rates."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in results.groupby("model", sort=False):
        group = group.sort_values("noise_rate")
        ax.errorbar(
            group["noise_rate"] * 100,
            group["mean_accuracy"],
            yerr=group["std_accuracy"],
            marker="o",
            capsize=3,
            label=model,
        )
    ax.set(
        title="Label-noise robustness (clean validation folds)",
        xlabel="Training-label flip rate (%)",
        ylabel="Mean validation accuracy (error bars: ±1 std)",
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_noise_robustness.png", dpi=300)
    plt.close(fig)
