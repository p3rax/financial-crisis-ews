import os

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "model_test_results.csv")


def ensure_fig_dir() -> None:
    """Create figures directory if it does not exist."""
    os.makedirs(FIG_DIR, exist_ok=True)


def _barplot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """Helper to create and save a bar plot for a given metric."""
    df_plot = df[["model", metric_col]].dropna(subset=[metric_col]).copy()
    if df_plot.empty:
        print(f"[viz] No valid values for {metric_col}. Skipping {filename}.")
        return

    plt.figure()
    plt.bar(df_plot["model"], df_plot[metric_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[viz] Saved: {out_path}")


def main() -> None:
    """
    Generate extra visual summaries from test results.

    Produces:
      - Bar plot of test ROC-AUC across models
      - Bar plot of test Average Precision across models
    """
    if not os.path.exists(RESULTS_PATH):
        print(f"[viz] No test results found at {RESULTS_PATH}. Skipping extra plots.")
        return

    df = pd.read_csv(RESULTS_PATH)

    required_cols = {"model", "roc_auc_test", "avg_precision_test"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[viz] Unexpected columns in {RESULTS_PATH}. Found: {list(df.columns)}")
        print("[viz] Skipping extra plots.")
        return

    ensure_fig_dir()

    _barplot(
        df=df,
        metric_col="roc_auc_test",
        title="Model performance on test set (ROC-AUC)",
        ylabel="Test ROC-AUC",
        filename="test_roc_auc_barplot.png",
    )

    _barplot(
        df=df,
        metric_col="avg_precision_test",
        title="Model performance on test set (Average Precision)",
        ylabel="Test Average Precision",
        filename="test_avg_precision_barplot.png",
    )


if __name__ == "__main__":
    main()