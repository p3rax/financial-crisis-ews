import os
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def ensure_data_dir() -> None:
    """Create raw data directory if it doesn't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def _raw_path(name: str) -> str:
    return os.path.join(RAW_DATA_DIR, f"{name}.csv")


def download_fred_series(series_id: str, start: str = "1990-01-01") -> Optional[pd.DataFrame]:
    """Download a time series from FRED using pandas_datareader."""
    try:
        df = pdr.get_data_fred(series_id, start=start)
        df = df.rename(columns={series_id: "value"})
        df.index = pd.to_datetime(df.index)
        return df
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        print(f"[download_data] Error downloading FRED {series_id}: {e}")
        return None


def download_yahoo_series(ticker: str, start: str = "1990-01-01") -> Optional[pd.DataFrame]:
    """Download a time series from Yahoo Finance using yfinance."""
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        df.index = pd.to_datetime(df.index)
        df = df[["Close"]].rename(columns={"Close": "value"})
        return df
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        print(f"[download_data] Error downloading Yahoo {ticker}: {e}")
        return None


def save_series(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame into data/raw/<name>.csv."""
    if df is None or df.empty:
        print(f"[download_data] Skipping {name}: empty dataframe.")
        return
    path = _raw_path(name)
    df.to_csv(path)
    print(f"[download_data] Saved {name} -> {path}")


def download_all(force: bool = False) -> None:
    """
    Download all datasets used in the EWS model.

    Parameters
    ----------
    force : bool
        If True, re-download even if raw CSV already exists.
    """
    ensure_data_dir()

    indicators_fred = {
        "HY_OAS": "BAMLH0A0HYM2",
        "IG_OAS": "BAMLC0A0CM",
        "STLFSI4": "STLFSI4",
        "T10Y2Y": "T10Y2Y",
    }

    indicators_yahoo = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
    }

    print("[download_data] Downloading FRED indicators...")
    for name, fred_id in indicators_fred.items():
        if not force and os.path.exists(_raw_path(name)):
            print(f"[download_data] Found existing {name}, skipping.")
            continue
        df = download_fred_series(fred_id)
        if df is not None:
            save_series(df, name)

    print("[download_data] Downloading Yahoo Finance series...")
    for name, ticker in indicators_yahoo.items():
        if not force and os.path.exists(_raw_path(name)):
            print(f"[download_data] Found existing {name}, skipping.")
            continue
        df = download_yahoo_series(ticker)
        if df is not None:
            save_series(df, name)

    print("[download_data] All downloads completed.")


if __name__ == "__main__":
    download_all()