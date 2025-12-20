import os
import numpy as np
import pytest

from src.modeling.train_models import load_ews_dataset, build_models, DATA_PATH
from src.modeling.explain_models import compute_shap_values_tree


def _find_window_with_both_classes(y, window_size: int = 300):
    y_arr = y.values
    for start in range(0, max(1, len(y_arr) - window_size)):
        y_win = y_arr[start : start + window_size]
        if len(np.unique(y_win)) >= 2:
            return start, start + window_size
    raise ValueError("Could not find a window with both classes in y.")


def test_shap_values_shape_and_validity():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Processed dataset not found. Run `python main.py` first.")

    X, y = load_ews_dataset()
    start, end = _find_window_with_both_classes(y, window_size=300)

    model = build_models(seed=42)["random_forest"]
    model.fit(X.iloc[start:end], y.iloc[start:end])

    _, shap_values, X_sample = compute_shap_values_tree(
        model,
        X.iloc[start:end],
        sample_size=100,
        seed=42,
    )

    assert shap_values.shape[0] == len(X_sample)
    assert shap_values.shape[1] == X_sample.shape[1]
    assert not np.isnan(shap_values).any()