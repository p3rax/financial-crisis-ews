import os
import pytest
import pandas as pd

from src.data.download_data import RAW_DATA_DIR


def test_raw_data_files_exist():
    if not os.path.exists(RAW_DATA_DIR):
        pytest.skip("data/raw not found. Run `python main.py` to download raw data.")

    expected_files = [
        "HY_OAS.csv",
        "IG_OAS.csv",
        "STLFSI4.csv",
        "T10Y2Y.csv",
        "VIX.csv",
        "DXY.csv",
    ]

    for fname in expected_files:
        path = os.path.join(RAW_DATA_DIR, fname)
        assert os.path.exists(path), f"Missing raw data file: {fname}"


def test_raw_data_format():
    if not os.path.exists(RAW_DATA_DIR):
        pytest.skip("data/raw not found. Run `python main.py` to download raw data.")

    path = os.path.join(RAW_DATA_DIR, "STLFSI4.csv")
    df = pd.read_csv(path)

    assert not df.empty