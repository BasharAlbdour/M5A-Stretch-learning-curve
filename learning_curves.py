import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder , PolynomialFeatures
from sklearn.linear_model import LogisticRegression


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_data(filepath="data/telecom_churn.csv"):

    df=pd.read_csv(filepath)
    X=df.drop(columns=['churned','customer_id'])
    y=df['churned']
    return X,y

def build_pipeline(C=1.0):
    preprocessor=ColumnTransformer(
        transformers=[
            ('num',StandardScaler(),NUMERIC_FEATURES),
            ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),CATEGORICAL_FEATURES)    
        ])
    model = LogisticRegression(C=C,random_state=42,max_iter=1000,class_weight="balanced")
    
    return Pipeline([('preprocessor', preprocessor), ('model', model)])

def build_pipeline_poly(C=1.0, degree=2):
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])

    model = LogisticRegression(
        C=C,
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def compute_learning_curves(pipeline, X, y, scoring="f1", cv=5, n_splits=10):
    train_sizes = np.linspace(0.1, 1.0, n_splits)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        train_sizes=train_sizes,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )
    
    return train_sizes_abs, train_scores, val_scores

def plot_learning_curves(train_sizes, train_scores, val_scores, title="Learning Curve"):
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training Score')
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2, color='steelblue')

    plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation Score')
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2, color='darkorange')

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.close()
    print("Plot saved to learning_curve.png")

def plot_comparison(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Compute curves for both models
    print("Computing baseline curves...")
    ts1, tr1, v1 = learning_curve(
        build_pipeline(C=1.0), X, y,
        train_sizes=train_sizes, cv=skf, scoring="f1", n_jobs=-1
    )

    print("Computing polynomial curves...")
    ts2, tr2, v2 = learning_curve(
        build_pipeline_poly(C=1.0, degree=2), X, y,
        train_sizes=train_sizes, cv=skf, scoring="f1", n_jobs=-1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, ts, tr, v, title in zip(
        axes,
        [ts1, ts2],
        [tr1, tr2],
        [v1, v2],
        ["Logistic Regression (baseline)", "Logistic Regression + Polynomial Features (degree=2)"]
    ):
        tr_mean, tr_std = tr.mean(axis=1), tr.std(axis=1)
        v_mean, v_std = v.mean(axis=1), v.std(axis=1)

        ax.plot(ts, tr_mean, 'o-', color='steelblue', label='Training Score')
        ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2, color='steelblue')

        ax.plot(ts, v_mean, 'o-', color='darkorange', label='Validation Score')
        ax.fill_between(ts, v_mean - v_std, v_mean + v_std, alpha=0.2, color='darkorange')

        ax.set_title(title)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("F1 Score")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bias-Variance Comparison: Baseline vs Polynomial Features", fontsize=13)
    plt.tight_layout()
    plt.savefig("comparison_curve.png", dpi=150)
    plt.close()
    print("Saved to comparison_curve.png")

Analysis= """
Scoring Metric Justification:
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

     
if __name__ == "__main__":
    X, y = load_data()
    print(f"Data: {X.shape[0]} rows | Churn rate: {y.mean():.2%}")

    pipeline = build_pipeline(C=1.0)

    print("Computing learning curves...")
    train_sizes, train_scores, val_scores = compute_learning_curves(pipeline, X, y)

    print(f"Final training F1:   {train_scores.mean(axis=1)[-1]:.4f}")
    print(f"Final validation F1: {val_scores.mean(axis=1)[-1]:.4f}")
    print(f"Gap: {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:.4f}")

    plot_learning_curves(train_sizes, train_scores, val_scores,
                         title="Logistic Regression Learning Curve (F1, class_weight=balanced)")
    print("\nGenerating complexity comparison plot...")
    plot_comparison(X, y)
    print(Analysis)