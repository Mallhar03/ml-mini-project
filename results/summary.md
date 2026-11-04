# Results summary

All cross-validation and tuning used the stratified training split only. The hold-out set was evaluated once after model selection; robustness experiments use a separate validation split drawn only from the training partition.

- Training split: 455 rows × 30 features.
- Training labels: 285 benign (62.64%), 170 malignant (37.36%).
- Positive class: malignant (`1`); recall, precision, and F1 refer to malignant.

## Model comparison

| Model | CV accuracy (mean ± std) | CV malignant recall | CV F1 | CV ROC-AUC | Hold-out accuracy | Hold-out malignant recall | Hold-out F1 | Hold-out ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdaBoost | 0.9659 ± 0.0193 | 0.9294 | 0.9528 | 0.9912 | 0.9737 | 0.9286 | 0.9630 | 0.9858 |
| Gradient boosting | 0.9580 ± 0.0231 | 0.9312 | 0.9429 | 0.9893 | 0.9649 | 0.9048 | 0.9500 | 0.9934 |
| Random forest | 0.9571 ± 0.0238 | 0.9288 | 0.9415 | 0.9875 | 0.9737 | 0.9286 | 0.9630 | 0.9929 |
| Bagging | 0.9503 ± 0.0264 | 0.9247 | 0.9326 | 0.9833 | 0.9737 | 0.9286 | 0.9630 | 0.9911 |
| Decision tree | 0.9202 ± 0.0289 | 0.8982 | 0.8933 | 0.9158 | 0.9298 | 0.9048 | 0.9048 | 0.9246 |
| Decision stump | 0.9024 ± 0.0294 | 0.8165 | 0.8615 | 0.8851 | 0.8772 | 0.7143 | 0.8108 | 0.8433 |
| Most frequent | 0.6264 ± 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6316 | 0.0000 | 0.0000 | 0.5000 |

## Statistical comparisons

- **Random forest vs AdaBoost:** mean paired accuracy difference -0.0088; corrected t-test p=0.3735 (not significant at α=0.05); secondary Wilcoxon p=0.00205. The tests disagree here; the dependence-corrected result is the primary interpretation.
- **Best ensemble vs Decision tree:** mean paired accuracy difference +0.0457; corrected t-test p=0.0032 (significant at α=0.05); secondary Wilcoxon p=3.037e-09.

## Findings (RQ1–RQ4)

- **RQ1:** Best repeated-CV accuracy was AdaBoost at 0.9659 ± 0.0193; the single full tree was 0.9202 ± 0.0289. Best ensemble vs. the single tree, on repeated CV, differed by accuracy +0.0457, malignant recall +0.0312, F1 +0.0595, ROC-AUC +0.0754.
- **RQ1 (hold-out):** Bagging, Random forest, AdaBoost tied for highest one-time test accuracy (0.9737); malignant recall among the tied models was 0.9286.
- **RQ2:** See corrected resampled t-test and secondary Wilcoxon results above; the corrected test is the primary significance decision.
- **RQ3:** Five-fold mean accuracy changed from Bagging 0.9209 at 1 to 0.9582 at 300; Random forest 0.9143 at 1 to 0.9604 at 300; AdaBoost 0.9011 at 1 to 0.9714 at 300; Gradient boosting 0.6264 at 1 to 0.9692 at 300 as ensemble size increased from 1 to 300. See `fig06_ensemble_size.png` and `ensemble_size.csv` for all requested sizes and standard deviations.
- **RQ4:** The mean 0%-to-30% accuracy drop was 0.1055 for the bagging family and 0.0835 for the boosting family. Thus, in this run, boosting was more robust on average; Bagging had the largest individual ensemble drop (0.1143). The comparison includes both members of each family and does not assume AdaBoost must be most sensitive. Scores use a clean, fixed stratified validation subset of the training data over five independent noise seeds. For a controlled and laptop-friendly noise experiment, all ensemble families use 50 estimators while retaining their selected non-size settings.
- **Bias–variance:** The full tree scored 1.0000 on training and 0.9202 in CV (gap 0.0798), consistent with overfitting. Bagging kept training accuracy at 1.0000 but narrowed the gap to 0.0497. The stump's training/CV accuracies were 0.9297/0.9024, consistent with underfitting; AdaBoost raised these to 0.9978/0.9659.

## Reproducibility

Run `python run_all.py`; package versions are recorded in `environment.txt`. Detailed metrics, per-fold tables, classification reports, and figures are saved under `results/tables/` and `results/figures/`.
