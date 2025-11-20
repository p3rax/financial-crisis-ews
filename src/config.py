from pathlib import Path

# === Project root ===
# Automatically detect the root of the project, regardless of where Python is executed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# === Data paths ===
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# === Models and results ===
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# === FRED series codes ===
# Official FRED identifiers for the macro-financial time series.
FRED_SERIES = {
    "HY_OAS": "BAMLH0A0HYM2",   # High Yield Option-Adjusted Spread
    "IG_OAS": "BAMLC0A0CM",     # Investment Grade Option-Adjusted Spread
    "STLFSI4": "STLFSI4",       # St. Louis Fed Financial Stress Index
    "T10Y2Y": "T10Y2Y",         # 10-Year minus 2-Year Treasury Yield
}

# === Yahoo Finance tickers ===
YAHOO_TICKERS = {
    "VIX": "^VIX",              # CBOE Volatility Index
    # TODO: verify correct ticker for the MOVE index
    "MOVE": "^MOVE",
    "DXY": "DX-Y.NYB",          # US Dollar Index
}

# === Time frequency settings ===
# Use weekly data sampled on Fridays.
WEEKLY_FREQ = "W-FRI"

# === Label / target configuration ===
# These columns will be created during preprocessing.
LABEL_COLUMNS = ["stress_now", "stress_next4w"]

# Main target for classification models.
TARGET_COLUMN = "stress_next4w"