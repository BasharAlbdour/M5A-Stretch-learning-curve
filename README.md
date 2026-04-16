## Analysis
# Scoring Metric Justification:
F1 score was chosen over accuracy because the telecom churn dataset is heavily
imbalanced (16.27% churn rate). A model that always predicts "no churn" would
score 83.7% accuracy while catching zero churners — making accuracy a completely
misleading metric. F1 balances precision and recall, which makes it a meaningful
signal for minority-class detection like churn identification.

1. High bias or high variance?
   The baseline logistic regression shows a clear high-bias signature. Both
   training and validation F1 scores converge at a low value (~0.375 and ~0.351
   respectively) with a tiny final gap of only 0.024. The model is drawing a
   straight decision boundary that cannot capture conditional patterns in the
   data — for example, how tenure interacts with contract type to predict churn.
   No matter how much data you give it, logistic regression can only draw that
   one straight boundary, so both curves plateau early and stay low.

2. Would collecting more data help?
   No. The validation curve flattens around 400–600 training samples and shows
   no meaningful improvement beyond that point. The bottleneck is model capacity,
   not data quantity — adding more rows to a model that is already too simple to
   represent the true patterns will not move the validation F1. This is the
   reading's definition of high bias exactly: more data will not help because
   the model is fundamentally limited by its assumptions.

3. Would increasing model complexity help?
   Not through polynomial features. Adding degree=2 polynomial features expanded
   the feature space from 11 to 77+ columns while the number of training rows
   stayed the same (~1200). The model gained enough flexibility to start
   memorizing noise — training score stayed higher (~0.39) while validation
   dropped below the baseline (~0.33). This is the worst of both worlds: still
   not flexible enough to capture the real patterns (still high bias) but
   flexible enough to start fitting noise (added variance). The validation F1
   ending up lower than the baseline confirms that polynomial features made
   the bias-variance tradeoff worse without improving either side meaningfully.
   The right kind of complexity is a smarter model architecture — tree-based
   models like Random Forest ask a sequence of conditional yes/no questions
   about the existing features, capturing nonlinear interactions natively
   without exploding the feature space.

4. Recommended next step:
   Switch to a tree-based model such as Random Forest or Gradient Boosting.
   These models have higher capacity by design, handle nonlinear feature
   interactions like tenure × contract type natively, and should push validation
   F1 meaningfully above the ~0.35 ceiling that logistic regression has hit.
   This is exactly what Module 5 Week B addresses — and the learning curves
   here make the case for why it is the necessary next step."""
