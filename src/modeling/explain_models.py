import os
import random
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.modeling.train_models import load_ews_dataset, MODELS_DIR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
TEST_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "model_test_results.csv")


def ensure_figure_dir() -> None:
    """Create figure directory if it does not exist."""
    os.makedirs(FIG_DIR, exist_ok=True)


def load_trimmed_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the EWS dataset and trim it so that it starts from the first positive event
    in the target. Same trimming logic used in training/evaluation.
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

    print(f"[shap] Trimmed dataset start date: {first_pos_date}")
    print(f"[shap] Shapes for SHAP: X={X.shape}, y={y.shape}")
    print(f"[shap] Positive class share (trimmed): {y.mean():.3f}")

    return X, y


def pick_best_model_name(
    default: str = "random_forest",
    metric_col: str = "avg_precision_test",
) -> str:
    """
    Pick the best model from results/tables/model_test_results.csv.
    Falls back to `default` if the file is missing or metrics are not available.
    """
    if not os.path.exists(TEST_RESULTS_PATH):
        print(f"[shap] Test results not found at {TEST_RESULTS_PATH}. Using default={default}.")
        return default

    df = pd.read_csv(TEST_RESULTS_PATH)
    if "model" not in df.columns or metric_col not in df.columns:
        print(f"[shap] Missing columns in {TEST_RESULTS_PATH}. Using default={default}.")
        return default

    df_valid = df.dropna(subset=[metric_col])
    if df_valid.empty:
        print(f"[shap] No valid '{metric_col}' values. Using default={default}.")
        return default

    best_idx = df_valid[metric_col].idxmax()
    best_model = str(df_valid.loc[best_idx, "model"])
    print(f"[shap] Selected best model from test results: {best_model} (by {metric_col})")
    return best_model


def load_model(model_name: str):
    """Load a trained model saved by train_models.py."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"[shap] Loaded model: {model_name} from {model_path}")
    return model


def compute_shap_values_tree(
    model,
    X: pd.DataFrame,
    sample_size: int = 400,
    seed: int = 42,
):
    """
    Compute SHAP values for tree-based models using TreeExplainer.

    Notes:
    - SHAP can return a list (per class) or a 3D array (n, p, classes).
    - We always return the SHAP values for the positive class (1) in a 2D array (n, p).
    """
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X

    print(f"[shap] Computing SHAP values on {len(X_sample)} samples...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    return explainer, shap_values_pos, X_sample


def plot_shap_summary(shap_values_pos, X_sample: pd.DataFrame, model_name: str) -> None:
    """SHAP summary (beeswarm) plot."""
    ensure_figure_dir()
    plt.figure()
    shap.summary_plot(shap_values_pos, X_sample, show=False)
    out = os.path.join(FIG_DIR, f"shap_summary_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[shap] Saved: {out}")


def plot_shap_bar(shap_values_pos, X_sample: pd.DataFrame, model_name: str) -> None:
    """SHAP bar plot (mean |SHAP|)."""
    ensure_figure_dir()
    plt.figure()
    shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", show=False)
    out = os.path.join(FIG_DIR, f"shap_bar_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[shap] Saved: {out}")


def plot_shap_dependence_top_feature(
    shap_values_pos, X_sample: pd.DataFrame, model_name: str
) -> None:
    """Dependence plot for the most important feature by mean |SHAP|."""
    ensure_figure_dir()

    shap_abs_mean = np.abs(shap_values_pos).mean(axis=0)
    top_idx = int(np.argmax(shap_abs_mean))
    top_feature = X_sample.columns[top_idx]
    print(f"[shap] Top feature by mean |SHAP|: {top_feature}")

    x = X_sample.iloc[:, top_idx].values
    y = shap_values_pos[:, top_idx]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(top_feature)
    plt.ylabel(f"SHAP value for {top_feature}")
    plt.title(f"Dependence plot for {top_feature} ({model_name})")

    out = os.path.join(FIG_DIR, f"shap_dependence_{top_feature}_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[shap] Saved: {out}")


def main(seed: int = 42) -> None:
    """
    Explain the best EWS model using SHAP.

    Steps:
      1) Load trimmed dataset (same logic as train/eval).
      2) Pick best model from test results (fallback to random_forest).
      3) Load the saved model from models/.
      4) Compute SHAP values (TreeExplainer) on a sample.
      5) Save summary, bar and dependence plots.
    """
    random.seed(seed)
    np.random.seed(seed)

    ensure_figure_dir()

    X, _ = load_trimmed_dataset()

    model_name = pick_best_model_name(default="random_forest")

    if model_name not in {"random_forest", "xgboost"}:
        print(
            f"[shap] Selected model '{model_name}' is not tree-based. Falling back to random_forest."
        )
        model_name = "random_forest"

    model = load_model(model_name=model_name)

    _, shap_values_pos, X_sample = compute_shap_values_tree(
        model, X, sample_size=400, seed=seed
    )

    plot_shap_summary(shap_values_pos, X_sample, model_name=model_name)
    plot_shap_bar(shap_values_pos, X_sample, model_name=model_name)
    plot_shap_dependence_top_feature(shap_values_pos, X_sample, model_name=model_name)

    print("\n[shap] SHAP explainability finished.")


if __name__ == "__main__":
    main(seed=42)