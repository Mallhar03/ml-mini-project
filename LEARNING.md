# Learning notes

## Bootstrap sampling

Bagging creates each training set by sampling the original data **with
replacement**. Some observations appear more than once and some are left out.
Separate trees are fit to these bootstrap samples in parallel.

## Why averaging reduces variance

A deep tree can change substantially when its training observations change.
Bagging averages predictions from trees trained on different bootstrap samples,
so tree-specific fluctuations tend to cancel. The reduction is strongest when
the individual trees are not perfectly correlated.

## AdaBoost sample weights

AdaBoost fits learners sequentially and increases the weights of observations
misclassified by the current ensemble. For a learner with weighted error
\(\epsilon_t\), its vote weight is
\(\alpha_t = \frac{1}{2}\ln((1-\epsilon_t)/\epsilon_t)\) in the binary
exponential-loss formulation. A learner with lower error receives a larger
vote; repeatedly difficult or mislabeled samples can gain influence.

## Gradient boosting

Gradient boosting adds small trees sequentially. Each new tree approximates
the negative gradient of the chosen loss with respect to current predictions;
for squared-error regression this is the residual. A learning rate shrinks
each update, trading slower fitting for regularization.

## Repeated CV and corrected testing

Repeated stratified CV evaluates models on many shared train/validation splits,
so mean and standard deviation show performance and split sensitivity. Those
fold scores are not independent because training sets overlap. The
Nadeau–Bengio corrected resampled t-test inflates the variance estimate for
that dependence; an uncorrected paired t-test can overstate evidence.
