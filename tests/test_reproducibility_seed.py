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


def test_random_forest_reproducible_with_seed():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Processed dataset not found. Run `python main.py` first.")

    X, y = load_ews_dataset()
    start, end = _find_window_with_both_classes(y, window_size=300)

    m1 = build_models(seed=42)["random_forest"]
    m2 = build_models(seed=42)["random_forest"]

    m1.fit(X.iloc[start:end], y.iloc[start:end])
    m2.fit(X.iloc[start:end], y.iloc[start:end])

    X_pred = X.iloc[end : end + 50]
    p1 = m1.predict_proba(X_pred)[:, 1]
    p2 = m2.predict_proba(X_pred)[:, 1]

    assert np.allclose(p1, p2, atol=0.0)