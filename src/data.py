"""Dataset loading, label convention, splitting, and concise EDA outputs."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import SEED, TEST_SIZE, RESULTS


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load WDBC and split it while keeping the test set out of model selection."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    # sklearn encodes malignant as 0; make malignant the positive class (1).
    labels = (dataset.target == 0).astype(int).rename("malignant")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    return X_train, X_test, y_train, y_test


def save_eda(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    """Save training-only descriptive statistics and return dataset facts."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    X_train.describe().T.to_csv(RESULTS / "tables" / "summary_statistics.csv")
    class_counts = y_train.value_counts().sort_index()
    class_percentages = (class_counts / len(y_train) * 100).round(2)
    return {
        "training_shape": X_train.shape,
        "class_counts": {
            "benign": int(class_counts.get(0, 0)),
            "malignant": int(class_counts.get(1, 0)),
        },
        "class_percentages": {
            "benign": float(class_percentages.get(0, 0.0)),
            "malignant": float(class_percentages.get(1, 0.0)),
        },
    }
