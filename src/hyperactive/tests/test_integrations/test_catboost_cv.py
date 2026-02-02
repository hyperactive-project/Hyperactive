import numpy as np
import pytest
from catboost import Pool
from hyperactive.experiment.integrations import CatBoostCvExperiment

@pytest.fixture
def dummy_binary_pool():
    """Create a small random binary classification dataset as CatBoost Pool."""
    np.random.seed(42)
    X = np.random.rand(400, 8)
    y = np.random.randint(0, 2, 400)
    return Pool(data=X, label=y)

def test_catboost_cv_runs_and_returns_valid_score(dummy_binary_pool):
    """Basic sanity test: ensure CatBoostCvExperiment runs cv() correctly."""
    exp = CatBoostCvExperiment(
        pool=dummy_binary_pool,
        metric="Logloss",
        fold_count=4,
        iterations=50,
        early_stopping_rounds=10,
    )

    params = {
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 3.0,
    }

    raw_score, metadata = exp.evaluate(params)
    signed_score, _ = exp.score(params)

    assert isinstance(raw_score, float)
    assert 0 < raw_score < 1, f"Logloss out of expected range: {raw_score:.4f}"
    assert signed_score < 0, "Signed score should be negative (lower is better)"
    assert exp.get_tag("property:higher_or_lower_is_better") == "lower"

    assert isinstance(metadata, dict)
    assert "cv_results" in metadata
    assert "target_column" in metadata
    assert metadata["target_column"] == "test-Logloss-mean"
    assert "final_iteration" in metadata
    assert metadata["final_iteration"] > 0
    assert "used_loss_function" in metadata
    assert metadata["used_loss_function"] == "Logloss"

def test_catboost_cv_with_auc(dummy_binary_pool):
    """Test AUC metric with Logloss objective (shows metric vs loss separation)."""
    exp = CatBoostCvExperiment(
        pool=dummy_binary_pool,
        loss_function="Logloss",          
        metric="AUC",                       
        fold_count=3,
        iterations=30,
    )

    params = {"learning_rate": 0.05, "depth": 4}

    raw_score, metadata = exp.evaluate(params)
    signed_score, _ = exp.score(params)

    # AUC on random data is noisy, but typically around 0.5 ± 0.2–0.3
    assert 0.25 < raw_score < 0.75, f"Unexpected AUC on random data: {raw_score:.4f}"

    # Optimizing Logloss (lower-better) → tag is "lower", signed_score is negative
    assert signed_score < 0, f"Signed score should be negative, got {signed_score:.4f}"

    assert exp.get_tag("property:higher_or_lower_is_better") == "lower"
    assert "used_loss_function" in metadata
    assert metadata["used_loss_function"] == "Logloss"