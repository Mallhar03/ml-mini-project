"""Reproduce the complete bagging-vs-boosting experiment."""

from datetime import datetime, timezone
import platform

import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn

from src.config import FIGURES, RESULTS, TABLES
from src.data import load_and_split, save_eda
from src.evaluate import (
    compare_significance,
    final_holdout_evaluation,
    fitted_model_map,
    repeated_cv_comparison,
    tune_ensembles,
)
from src.experiments import (
    bias_variance_experiment,
    ensemble_size_experiment,
    noise_robustness_experiment,
)
from src.plots import (
    plot_cv_boxplot,
    plot_ensemble_size,
    plot_eda,
    plot_holdout,
    plot_noise_robustness,
)


def write_environment() -> None:
    versions = {
        "Python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "NumPy": numpy.__version__,
        "pandas": pandas.__version__,
        "SciPy": scipy.__version__,
        "Matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "Generated (UTC)": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "environment.txt").write_text(
        "\n".join(f"{name}: {version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )


def write_summary(
    eda: dict[str, object],
    cv_table: pandas.DataFrame,
    test_table: pandas.DataFrame,
    significance: pandas.DataFrame,
    size_table: pandas.DataFrame,
    bias_variance: pandas.DataFrame,
    noise_table: pandas.DataFrame,
) -> None:
    top = cv_table.iloc[0]
    tree_cv = cv_table.set_index("model").loc["Decision tree"]
    ensemble_names = ["Bagging", "Random forest", "AdaBoost", "Gradient boosting"]
    best_ensemble = cv_table[cv_table["model"].isin(ensemble_names)].iloc[0]
    max_test_accuracy = test_table["accuracy"].max()
    best_test_models = test_table.loc[
        test_table["accuracy"] == max_test_accuracy, "model"
    ].tolist()
    best_test_recall = test_table.loc[
        test_table["accuracy"] == max_test_accuracy, "recall_malignant"
    ].max()
    noise_summary = noise_table.pivot(
        index="model", columns="noise_rate", values="mean_accuracy"
    )
    noise_drops = noise_summary[0.0] - noise_summary[0.3]
    family_drops = {
        "Bagging": float(noise_drops[["Bagging", "Random forest"]].mean()),
        "Boosting": float(noise_drops[["AdaBoost", "Gradient boosting"]].mean()),
    }
    less_robust_family = max(family_drops, key=family_drops.get)
    more_robust_family = min(family_drops, key=family_drops.get)
    largest_ensemble_drop = noise_drops[ensemble_names].idxmax()
    largest_ensemble_drop_value = float(noise_drops[largest_ensemble_drop])

    tree_row = cv_table.set_index("model").loc["Decision tree"]
    rq1_differences = {
        "accuracy": best_ensemble["accuracy_mean"] - tree_row["accuracy_mean"],
        "malignant recall": best_ensemble["recall_mean"] - tree_row["recall_mean"],
        "F1": best_ensemble["f1_mean"] - tree_row["f1_mean"],
        "ROC-AUC": best_ensemble["roc_auc_mean"] - tree_row["roc_auc_mean"],
    }
    bias = bias_variance.set_index("model")
    tree_bias = bias.loc["Decision tree"]
    bagging_bias = bias.loc["Bagging"]
    stump_bias = bias.loc["Decision stump"]
    adaboost_bias = bias.loc["AdaBoost"]
    size_lines = []
    for model_name, group in size_table.groupby("model", sort=False):
        group = group.sort_values("n_estimators")
        first, last = group.iloc[0], group.iloc[-1]
        size_lines.append(
            f"{model_name} {first['mean_accuracy']:.4f} at {int(first['n_estimators'])} "
            f"to {last['mean_accuracy']:.4f} at {int(last['n_estimators'])}"
        )
    comparison_lines = [
        "| Model | CV accuracy (mean ± std) | CV malignant recall | CV F1 | CV ROC-AUC | Hold-out accuracy | Hold-out malignant recall | Hold-out F1 | Hold-out ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test_by_model = test_table.set_index("model")
    for _, row in cv_table.iterrows():
        test_row = test_by_model.loc[row["model"]]
        comparison_lines.append(
            f"| {row['model']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} "
            f"| {row['recall_mean']:.4f} | {row['f1_mean']:.4f} | {row['roc_auc_mean']:.4f} "
            f"| {test_row['accuracy']:.4f} | {test_row['recall_malignant']:.4f} "
            f"| {test_row['f1_malignant']:.4f} | {test_row['roc_auc']:.4f} |"
        )
    test_comparisons = []
    for _, row in significance.iterrows():
        conclusion = "significant" if row["significant_at_0.05"] else "not significant"
        test_comparisons.append(
            f"- **{row['comparison']}:** mean paired accuracy difference "
            f"{row['mean_accuracy_difference']:+.4f}; corrected t-test p="
            f"{row['nadeau_bengio_p_value']:.4g} ({conclusion} at α=0.05); "
            f"secondary Wilcoxon p={row['wilcoxon_p_value_secondary']:.4g}. "
            + (
                "The tests disagree here; the dependence-corrected result is the primary interpretation."
                if row["nadeau_bengio_p_value"] >= 0.05
                and row["wilcoxon_p_value_secondary"] < 0.05
                else ""
            )
        )
    lines = [
        "# Results summary",
        "",
        "All cross-validation and tuning used the stratified training split only. "
        "The hold-out set was evaluated once after model selection; robustness experiments "
        "use a separate validation split drawn only from the training partition.",
        "",
        f"- Training split: {eda['training_shape'][0]} rows × {eda['training_shape'][1]} features.",
        f"- Training labels: {eda['class_counts']['benign']} benign "
        f"({eda['class_percentages']['benign']:.2f}%), "
        f"{eda['class_counts']['malignant']} malignant "
        f"({eda['class_percentages']['malignant']:.2f}%).",
        "- Positive class: malignant (`1`); recall, precision, and F1 refer to malignant.",
        "",
        "## Model comparison",
        "",
        *comparison_lines,
        "",
        "## Statistical comparisons",
        "",
        *test_comparisons,
        "",
        "## Findings (RQ1–RQ4)",
        "",
        f"- **RQ1:** Best repeated-CV accuracy was {top['model']} at "
        f"{top['accuracy_mean']:.4f} ± {top['accuracy_std']:.4f}; the single full tree "
        f"was {tree_cv['accuracy_mean']:.4f} ± {tree_cv['accuracy_std']:.4f}. "
        f"Best ensemble vs. the single tree, on repeated CV, differed by "
        + ", ".join(
            f"{metric} {value:+.4f}" for metric, value in rq1_differences.items()
        )
        + ".",
        f"- **RQ1 (hold-out):** {', '.join(best_test_models)} tied for highest one-time "
        f"test accuracy ({max_test_accuracy:.4f}); malignant recall among the tied "
        f"models was {best_test_recall:.4f}.",
        f"- **RQ2:** See corrected resampled t-test and secondary Wilcoxon results above; "
        "the corrected test is the primary significance decision.",
        "- **RQ3:** Five-fold mean accuracy changed from "
        + "; ".join(size_lines)
        + " as ensemble size increased from 1 to 300. See `fig06_ensemble_size.png` "
        "and `ensemble_size.csv` for all requested sizes and standard deviations.",
        f"- **RQ4:** The mean 0%-to-30% accuracy drop was "
        f"{family_drops['Bagging']:.4f} for the bagging family and "
        f"{family_drops['Boosting']:.4f} for the boosting family. Thus, in this run, "
        f"{more_robust_family.lower()} was more robust on average; "
        f"{largest_ensemble_drop} had the largest individual ensemble drop "
        f"({largest_ensemble_drop_value:.4f}). The comparison includes both members "
        "of each family and does not assume AdaBoost must be most sensitive. Scores "
        "use a clean, fixed "
        "stratified validation subset of the training data over five independent "
        "noise seeds. For a controlled and "
        "laptop-friendly noise experiment, all ensemble families use 50 estimators "
        "while retaining their selected non-size settings.",
        f"- **Bias–variance:** The full tree scored {tree_bias['training_accuracy']:.4f} "
        f"on training and {tree_bias['mean_cv_accuracy']:.4f} in CV (gap "
        f"{tree_bias['train_minus_cv_gap']:.4f}), consistent with overfitting. Bagging "
        f"kept training accuracy at {bagging_bias['training_accuracy']:.4f} but narrowed "
        f"the gap to {bagging_bias['train_minus_cv_gap']:.4f}. The stump's training/CV "
        f"accuracies were {stump_bias['training_accuracy']:.4f}/"
        f"{stump_bias['mean_cv_accuracy']:.4f}, consistent with underfitting; AdaBoost "
        f"raised these to {adaboost_bias['training_accuracy']:.4f}/"
        f"{adaboost_bias['mean_cv_accuracy']:.4f}.",
        "",
        "## Reproducibility",
        "",
        "Run `python run_all.py`; package versions are recorded in `environment.txt`. "
        "Detailed metrics, per-fold tables, classification reports, and figures are "
        "saved under `results/tables/` and `results/figures/`.",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_environment()

    print("Step 1/6: loading data and creating training-only EDA")
    X_train, X_test, y_train, y_test = load_and_split()
    eda = save_eda(X_train, y_train)
    plot_eda(X_train, y_train)
    print(f"Training shape: {eda['training_shape']}; class counts: {eda['class_counts']}")

    print("\nStep 2/6: tuning ensembles on training data")
    best_params, _ = tune_ensembles(X_train, y_train)

    print("\nStep 3/6: repeated-CV comparison and significance tests")
    models = fitted_model_map(best_params)
    cv_table, cv_scores = repeated_cv_comparison(X_train, y_train, models)
    plot_cv_boxplot(
        pandas.DataFrame(
            [
                {"model": name, "accuracy": score}
                for name, values in cv_scores.items()
                for score in values
            ]
        )
    )
    significance = compare_significance(cv_scores, cv_table)
    print(cv_table[["model", "accuracy_mean", "accuracy_std", "recall_mean"]].to_string(index=False))

    print("\nStep 4/6: final one-time hold-out evaluation")
    test_table, predictions, probabilities = final_holdout_evaluation(
        X_train, y_train, X_test, y_test, fitted_model_map(best_params)
    )
    plot_holdout(y_test, predictions, probabilities)

    print("\nStep 5/6: ensemble-size, bias–variance, and label-noise experiments")
    size_table, tree_accuracy = ensemble_size_experiment(X_train, y_train, best_params)
    plot_ensemble_size(size_table, tree_accuracy)
    bias_variance = bias_variance_experiment(X_train, y_train, best_params, cv_scores)
    noise_table = noise_robustness_experiment(X_train, y_train, best_params)
    plot_noise_robustness(noise_table)

    print("\nStep 6/6: writing evidence-based summary")
    write_summary(
        eda, cv_table, test_table, significance, size_table, bias_variance, noise_table
    )
    print(f"Complete. Results saved under {RESULTS}")


if __name__ == "__main__":
    main()
