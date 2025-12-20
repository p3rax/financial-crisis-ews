import pandas as pd
from src.modeling.evaluate_models import time_based_train_test_split_by_date


def test_time_split_no_leakage():
    idx = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    X = pd.DataFrame({"a": range(20)}, index=idx)
    y = pd.Series([0]*10 + [1] + [0]*9, index=idx)

    X_train, X_test, y_train, y_test = time_based_train_test_split_by_date(
        X, y, test_start_date="2020-03-13"
    )

    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()