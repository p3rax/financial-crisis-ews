import os
import joblib
import random
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ews_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CV_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "model_cv_results.csv")


def ensure_models_dir() -> None:
    """Create models directory if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_ews_dataset(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed EWS dataset and split it into features X and target y.

    Target:
        - 'stress_next_4w' (binary early-warning label).

    Features:
        - all columns except STLFSI4, stress_today, and the target itself.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=[0])

    target_col = "stress_next_4w"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    drop_cols = ["STLFSI4", "stress_today", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    return X, y


def build_models(seed: int = 42) -> dict[str, object]:
    """
    Define the ML models:
    - Logistic Regression with standardization (Pipeline);
    - Random Forest;
    - XGBoost.
    """
    models: dict[str, object] = {}

    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    models["logistic_regression"] = lr_pipeline

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    models["random_forest"] = rf

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    models["xgboost"] = xgb

    return models


def get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities P(y=1) for a fitted model."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    raise ValueError("Model does not implement predict_proba.")


def evaluate_models_cv(
    X: pd.DataFrame, y: pd.Series, models: dict[str, object], n_splits: int = 5
) -> pd.DataFrame:
    """
    Evaluate each model using an expanding TimeSeriesSplit cross-validation.

    For each split:
      - fit on past data
      - evaluate on the next fold
      - if y_train has only one class -> skip
      - if y_test has only one class -> skip scoring (AUC undefined)

    Metrics:
      - ROC-AUC
      - Average Precision (PR-AUC)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records: list[dict] = []

    for name, base_model in models.items():
        print(f"\n=== Evaluating {name} with TimeSeriesSplit ===")

        fold_rocs: list[float] = []
        fold_aps: list[float] = []
        split_idx = 0

        for train_index, test_index in tscv.split(X):
            split_idx += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if y_train.nunique() < 2:
                print(
                    f"  - Split {split_idx}: skipped (y_train has only one class: "
                    f"{y_train.unique().tolist()})"
                )
                continue

            model = clone(base_model)
            model.fit(X_train, y_train)

            if y_test.nunique() < 2 or y_test.sum() == 0:
                print(
                    f"  - Split {split_idx}: skipped scoring (y_test has only one class: "
                    f"{y_test.unique().tolist()})"
                )
                continue

            y_prob = get_probabilities(model, X_test)
            roc = roc_auc_score(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)

            fold_rocs.append(roc)
            fold_aps.append(ap)

            print(f"  - Split {split_idx}: ROC-AUC={roc:.3f}, Average Precision={ap:.3f}")

        if len(fold_rocs) == 0:
            mean_roc = float("nan")
            mean_ap = float("nan")
            print("  -> No valid splits with both classes present.")
        else:
            mean_roc = float(np.mean(fold_rocs))
            mean_ap = float(np.mean(fold_aps))
            print(f"ROC-AUC (mean over valid folds): {mean_roc:.3f}")
            print(f"Average Precision (mean over valid folds): {mean_ap:.3f}")

        records.append(
            {
                "model": name,
                "roc_auc_mean": mean_roc,
                "avg_precision_mean": mean_ap,
                "n_valid_splits": len(fold_rocs),
            }
        )

    return pd.DataFrame.from_records(records)


def fit_and_save_models(X: pd.DataFrame, y: pd.Series, models: dict[str, object]) -> None:
    """Fit each model on the full dataset and save it to disk using joblib."""
    ensure_models_dir()

    for name, model in models.items():
        print(f"\n=== Fitting {name} on full dataset ===")
        model.fit(X, y)

        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")


def main(seed: int = 42) -> None:
    """End-to-end training script for the EWS models."""
    random.seed(seed)
    np.random.seed(seed)

    print("Loading EWS dataset...")
    X, y = load_ews_dataset()

    print(f"Original dataset shape: X={X.shape}, y={y.shape}")
    print(f"Original positive class share: {y.mean():.3f}")

    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        raise ValueError(
            "No positive events found in target. "
            "Try lowering threshold_k in preprocessing or changing the horizon."
        )

    first_pos_idx = int(pos_idx[0])
    first_pos_date = y.index[first_pos_idx]

    X = X.iloc[first_pos_idx:]
    y = y.iloc[first_pos_idx:]

    print(f"\nTrimmed dataset to start at first positive event: {first_pos_date}")
    print(f"New dataset shape: X={X.shape}, y={y.shape}")
    print(f"New positive class share: {y.mean():.3f}")

    models = build_models(seed=seed)

    print("\nRunning time-series cross-validation...")
    cv_results = evaluate_models_cv(X, y, models, n_splits=5)

    os.makedirs(os.path.dirname(CV_RESULTS_PATH), exist_ok=True)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCross-validation results saved to: {CV_RESULTS_PATH}")

    print("\nFitting models on full (trimmed) dataset and saving them...")
    fit_and_save_models(X, y, models)

    print("\nAll models trained and saved successfully.")


if __name__ == "__main__":
    main(seed=42)