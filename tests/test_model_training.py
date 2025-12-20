import os
import numpy as np
import pytest

from src.modeling.train_models import load_ews_dataset, build_models, DATA_PATH


def _find_window_with_both_classes(y, window_size: int = 300):
    y_arr = y.values
    for start in range(0, max(1, len(y_arr) - window_size)):
        y_win = y_arr[start : start + window_size]
        if len(np.unique(y_win)) >= 2:
            return start, start + window_size
    raise ValueError("Could not find a window with both classes in y.")


def test_model_training_runs():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Processed dataset not found. Run `python main.py` first.")

    X, y = load_ews_dataset()
    models = build_models()

    start, end = _find_window_with_both_classes(y, window_size=300)

    m = models["logistic_regression"]
    m.fit(X.iloc[start:end], y.iloc[start:end])

    assert hasattr(m, "predict_proba")