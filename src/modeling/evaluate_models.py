import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

from src.modeling.train_models import load_ews_dataset, build_models, DATA_PATH


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

# >>> Change this date if you want a different test period <<<
TEST_START_DATE = "2012-01-01"


def ensure_output_dirs() -> None:
    """Create output directories for figures and tables."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def load_trimmed_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the EWS dataset and trim it so that it starts from the first positive event
    in the target. This avoids very early periods with no systemic stress events at all.
    """
    X, y = load_ews_dataset()

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

    print(f"[eval] Trimmed dataset start date: {first_pos_date}")
    print(f"[eval] Trimmed shapes: X={X.shape}, y={y.shape}")
    print(f"[eval] Positive class share (trimmed): {y.mean():.3f}")

    return X, y


def time_based_train_test_split_by_date(
    X: pd.DataFrame, y: pd.Series, test_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a chronological train/test split based on a calendar date.
    All observations with index >= test_start_date go to the test set, the rest go to the training set.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    test_start_date_dt = pd.to_datetime(test_start_date)
    mask_test = X.index >= test_start_date_dt

    if mask_test.sum() == 0:
        raise ValueError("Test start date is after the end of the dataset.")
    if (~mask_test).sum() == 0:
        raise ValueError(
            "Test start date is before the start of the dataset (train would be empty)."
        )

    X_train = X.loc[~mask_test]
    y_train = y.loc[~mask_test]
    X_test = X.loc[mask_test]
    y_test = y.loc[mask_test]

    print(f"\n[eval] Train from {X_train.index[0]} to {X_train.index[-1]}")
    print(f"[eval] Test  from {X_test.index[0]} to {X_test.index[-1]}")
    print(
        f"[eval] Train y mean = {y_train.mean():.3f}, Test y mean = {y_test.mean():.3f}"
    )

    if y_test.sum() == 0:
        print(
            "[eval] WARNING: no positive events in test set. "
            "You may want to choose an earlier TEST_START_DATE."
        )

    return X_train, X_test, y_train, y_test


def get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities P(y=1) for a fitted model."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not implement predict_proba.")
    proba = model.predict_proba(X)
    return proba[:, 1]


def calibrate_on_train_only(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    n_splits: int = 5,
):
    """
    Calibrate probabilities using ONLY training data with TimeSeriesSplit.
    This avoids any leakage from the hold-out test set.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=tscv)
    calibrated.fit(X_train, y_train)
    return calibrated


def fit_models_on_train(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> Dict[str, object]:
    """Fit fresh models on the training set only (no test contamination)."""
    models = build_models(seed=seed)
    fitted: Dict[str, object] = {}

    for name, model in models.items():
        print(f"\n[eval] Fitting {name} on TRAIN only...")
        model.fit(X_train, y_train)
        fitted[name] = model

    return fitted


def evaluate_on_test_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> pd.DataFrame:
    """Evaluate each trained model on the hold-out test set (ROC-AUC and Average Precision)."""
    records = []

    for name, model in models.items():
        print(f"\n[eval] === Evaluating {name} on TEST set ===")
        y_prob = get_probabilities(model, X_test)

        if y_test.nunique() < 2 or y_test.sum() == 0:
            print("[eval] Test set has only one class; ROC/PR metrics are undefined.")
            roc = float("nan")
            ap = float("nan")
        else:
            roc = roc_auc_score(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            print(f"[eval] Test ROC-AUC: {roc:.3f}")
            print(f"[eval] Test Average Precision: {ap:.3f}")

        records.append(
            {
                "model": name,
                "roc_auc_test": roc,
                "avg_precision_test": ap,
            }
        )

    return pd.DataFrame.from_records(records)


def plot_roc_pr_curves(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> None:
    """Plot ROC and Precision-Recall curves for all models on the test set."""
    if y_test.nunique() < 2 or y_test.sum() == 0:
        print("[eval] Skipping ROC/PR plots: test set has only one class.")
        return

    plt.figure()
    for name, model in models.items():
        y_prob = get_probabilities(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves on test set")
    plt.legend()
    roc_path = os.path.join(FIG_DIR, "roc_curves_test.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"[eval] ROC curves saved to: {roc_path}")

    plt.figure()
    for name, model in models.items():
        y_prob = get_probabilities(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=name)

    baseline = float(y_test.mean())
    plt.hlines(baseline, 0, 1, linestyles="dashed", label="Baseline (class share)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves on test set")
    plt.legend()
    pr_path = os.path.join(FIG_DIR, "pr_curves_test.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"[eval] Precision-Recall curves saved to: {pr_path}")


def plot_timeline_best_model(
    best_model_name: str,
    model,
    X_all: pd.DataFrame,
    y_all: pd.Series,
) -> None:
    """Plot a timeline of STLFSI4 and predicted risk scores of the best model over the trimmed sample."""
    df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])

    y_prob_full = get_probabilities(model, X_all)

    stl = df_full.loc[X_all.index, "STLFSI4"]
    threshold = stl.mean() + 2.0 * stl.std()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(stl.index, stl.values, label="STLFSI4", alpha=0.8)
    ax1.axhline(threshold, color="red", linestyle="--", label="Stress threshold")
    ax1.set_ylabel("STLFSI4")

    ax2 = ax1.twinx()
    ax2.plot(X_all.index, y_prob_full, label=f"{best_model_name} risk score", alpha=0.8)
    ax2.set_ylabel("Predicted risk score")

    stress_events = (y_all == 1)
    ax1.scatter(
        X_all.index[stress_events],
        stl[stress_events],
        color="black",
        marker="x",
        label="Systemic stress (target=1)",
    )

    ax1.set_title(f"Systemic stress and predicted risk ({best_model_name})")
    fig.tight_layout()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    timeline_path = os.path.join(FIG_DIR, f"timeline_{best_model_name}.png")
    plt.savefig(timeline_path, bbox_inches="tight")
    plt.close()
    print(f"[eval] Timeline plot saved to: {timeline_path}")


def main(seed: int = 42) -> None:
    """End-to-end evaluation of EWS models on a hold-out test set."""
    random.seed(seed)
    np.random.seed(seed)

    ensure_output_dirs()

    X_all, y_all = load_trimmed_dataset()

    X_train, X_test, y_train, y_test = time_based_train_test_split_by_date(
        X_all, y_all, test_start_date=TEST_START_DATE
    )

    fitted_models = fit_models_on_train(X_train, y_train, seed=seed)

    test_results = evaluate_on_test_set(X_test, y_test, fitted_models)
    results_path = os.path.join(TABLES_DIR, "model_test_results.csv")
    test_results.to_csv(results_path, index=False)
    print(f"\n[eval] Test-set results saved to: {results_path}")
    print(test_results)

    plot_roc_pr_curves(X_test, y_test, fitted_models)

    if test_results["avg_precision_test"].notna().any():
        best_idx = test_results["avg_precision_test"].idxmax()
        best_model_name = str(test_results.loc[best_idx, "model"])
        print(f"\n[eval] Best model on test set (by Average Precision): {best_model_name}")

        plot_timeline_best_model(
            best_model_name=best_model_name,
            model=fitted_models[best_model_name],
            X_all=X_all,
            y_all=y_all,
        )

        if y_test.nunique() >= 2 and y_test.sum() > 0:
            base_model = fitted_models[best_model_name]

            cal_model = calibrate_on_train_only(
                base_model, X_train, y_train, method="sigmoid", n_splits=5
            )

            y_prob_base = get_probabilities(base_model, X_test)
            y_prob_cal = get_probabilities(cal_model, X_test)

            brier_base = brier_score_loss(y_test, y_prob_base)
            brier_cal = brier_score_loss(y_test, y_prob_cal)

            cal_table = pd.DataFrame(
                [
                    {
                        "model": best_model_name,
                        "brier_uncalibrated": brier_base,
                        "brier_calibrated": brier_cal,
                    }
                ]
            )

            cal_path = os.path.join(TABLES_DIR, "calibration_results.csv")
            cal_table.to_csv(cal_path, index=False)
            print(f"[eval] Calibration results saved to: {cal_path}")

            frac_pos_base, mean_pred_base = calibration_curve(
                y_test, y_prob_base, n_bins=10, strategy="quantile"
            )
            frac_pos_cal, mean_pred_cal = calibration_curve(
                y_test, y_prob_cal, n_bins=10, strategy="quantile"
            )

            plt.figure()
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            plt.plot(
                mean_pred_base,
                frac_pos_base,
                marker="o",
                label=f"{best_model_name} (uncalibrated)",
            )
            plt.plot(
                mean_pred_cal,
                frac_pos_cal,
                marker="o",
                label=f"{best_model_name} (calibrated)",
            )
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title(f"Calibration curve on test set ({best_model_name})")
            plt.legend()

            cal_fig_path = os.path.join(FIG_DIR, f"calibration_curve_{best_model_name}.png")
            plt.savefig(cal_fig_path, bbox_inches="tight")
            plt.close()
            print(f"[eval] Calibration curve saved to: {cal_fig_path}")
        else:
            print("[eval] Skipping calibration: test set has only one class or no positives.")

    else:
        print("[eval] No valid test metrics to select a best model.")


if __name__ == "__main__":
    main(seed=42)