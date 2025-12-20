from src.modeling.explain_models import pick_best_model_name


def test_pick_best_model_fallback(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: False)
    best = pick_best_model_name(default="random_forest")
    assert best == "random_forest"