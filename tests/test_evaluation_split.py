import pandas as pd
from src.modeling.evaluate_models import time_based_train_test_split_by_date


def test_time_split_is_chronological():
    idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    X = pd.DataFrame({"a": range(10)}, index=idx)
    y = pd.Series([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], index=idx)

    X_train, X_test, y_train, y_test = time_based_train_test_split_by_date(
        X, y, test_start_date="2020-02-14"
    )

    assert X_train.index.max() < X_test.index.min()
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)