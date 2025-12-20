import os
from typing import Tuple

import pandas as pd
import warnings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def ensure_processed_dir() -> None:
    """Create processed data directory if it does not exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def load_series(name: str, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Load a time series from data/raw/<name>.csv, parse dates safely, resample to weekly frequency,
    and rename the value column to <name>.
    """
    path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
            )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date"})
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    if "value" not in df.columns:
        if "Close" in df.columns:
            df["value"] = df["Close"]
        else:
            raise ValueError(
                f"'value' column not found in {path}. "
                f"Available columns: {df.columns.tolist()}"
            )

    df = df[["value"]]
    df_weekly = df.resample(freq).last()
    df_weekly = df_weekly.rename(columns={"value": name})
    return df_weekly


def build_merged_dataset(freq: str = "W-FRI") -> pd.DataFrame:
    """Load all time series and merge them into a single weekly DataFrame."""
    series_names = ["HY_OAS", "IG_OAS", "VIX", "DXY", "T10Y2Y", "STLFSI4"]

    frames = [load_series(name, freq=freq) for name in series_names]
    df_all = pd.concat(frames, axis=1).sort_index()

    n_missing_before = int(df_all.isna().sum().sum())
    df_all = df_all.ffill().dropna()
    n_missing_after = int(df_all.isna().sum().sum())

    print(f"[preprocess] Missing values before ffill/dropna: {n_missing_before}")
    print(f"[preprocess] Missing values after  ffill/dropna: {n_missing_after}")
    print(f"[preprocess] Final merged dataset shape: {df_all.shape}")

    return df_all


def add_stress_targets(
    df: pd.DataFrame,
    horizon_weeks: int = 4,
    threshold_k: float = 2.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Add systemic stress targets based on STLFSI4.

    Creates:
      - 'stress_today': 1 if STLFSI4 > mean + k * std, else 0
      - 'stress_next_<horizon_weeks>w': 1 if any stress_today occurs in next horizon_weeks
    """
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1")

    if "STLFSI4" not in df.columns:
        raise ValueError("STLFSI4 column is required to build stress targets.")

    stl = df["STLFSI4"]
    threshold = stl.mean() + threshold_k * stl.std()

    df = df.copy()
    df["stress_today"] = (stl > threshold).astype(int)

    future_stress = pd.concat(
        [df["stress_today"].shift(-i) for i in range(1, horizon_weeks + 1)],
        axis=1,
    )
    df[f"stress_next_{horizon_weeks}w"] = future_stress.max(axis=1)

    df = df.iloc[:-horizon_weeks]
    return df, threshold


def make_xy(
    df: pd.DataFrame,
    horizon_weeks: int = 4,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for modeling, excluding STLFSI4 and target helper columns."""
    target_col = f"stress_next_{horizon_weeks}w"
    drop_cols = ["STLFSI4", "stress_today", target_col]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    return X, y


def build_and_save_ews_dataset(
    freq: str = "W-FRI",
    horizon_weeks: int = 4,
    threshold_k: float = 2.0,
) -> pd.DataFrame:
    """Build the full EWS dataset and save it to data/processed/ews_dataset.csv."""
    ensure_processed_dir()

    df_all = build_merged_dataset(freq=freq)
    df_all, threshold = add_stress_targets(
        df_all, horizon_weeks=horizon_weeks, threshold_k=threshold_k
    )

    out_path = os.path.join(PROCESSED_DATA_DIR, "ews_dataset.csv")
    df_all.to_csv(out_path)

    print(f"[preprocess] EWS dataset saved to: {out_path}")
    print(f"[preprocess] STLFSI4 threshold (mean + {threshold_k} * std): {threshold:.4f}")
    print(f"[preprocess] Final observations: {len(df_all)}")

    return df_all


if __name__ == "__main__":
    build_and_save_ews_dataset()