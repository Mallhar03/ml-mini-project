# Bagging vs. Boosting for Breast Cancer Diagnosis

This mini-project compares a single decision tree and stump with bagging,
random forests, AdaBoost, and gradient boosting on scikit-learn's Wisconsin
Diagnostic Breast Cancer (WDBC) dataset. It uses only the listed Python
scientific stack; no dataset download is required.

## Reproduce the results

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_all.py
```

The pipeline records package versions, creates every table and figure, and
writes `results/summary.md`. The run may take a few minutes because it includes
repeated CV and label-noise trials.

## Labels and evaluation protocol

The original sklearn labels use `0 = malignant` and `1 = benign`. This project
remaps them to **`1 = malignant` (positive class)** and `0 = benign`. All
precision, recall, and F1 metrics refer to malignant; malignant recall is
important because a false negative means a missed cancer diagnosis.

A stratified 80/20 split uses `random_state=42`. The training data alone is used
for five-fold tuning, 5×10 repeated stratified CV, ensemble-size experiments,
bias–variance comparisons, and label-noise experiments. The label-noise
experiment uses a separate stratified 80/20 validation split from the training
partition; it flips labels only in that split's fitting portion and keeps its
validation portion clean. The test split is reserved for final evaluation.
Trees are not scaled. The corrected
Nadeau–Bengio resampled t-test is the primary significance test; Wilcoxon is
reported as a secondary check.

## Project layout

- `run_all.py`: single end-to-end entry point.
- `src/`: data, models, evaluation, experiments, plotting, and shared settings.
- `results/tables/`: metrics, parameters, tests, and experiment data.
- `results/figures/`: report-ready figures at 300 dpi.
- `LEARNING.md`: concise viva notes.

## References

- Breiman (1996), “Bagging Predictors.”
- Breiman (2001), “Random Forests.”
- Freund & Schapire (1997), “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.”
- Friedman (2001), “Greedy Function Approximation: A Gradient Boosting Machine.”
- Nadeau & Bengio (2003), “Inference for the Generalization Error.”
- Dietterich (2000), “An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees.”
- Street, Wolberg & Mangasarian (1993), “Nuclear Feature Extraction for Breast Tumor Diagnosis.”

## Results snapshot

The reproducible, data-derived comparison is generated at
[`results/summary.md`](results/summary.md), with versions in
[`results/environment.txt`](results/environment.txt). No performance numbers
are hard-coded in this README.
