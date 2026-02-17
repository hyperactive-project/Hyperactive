"""Test skforecast integration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from hyperactive.integrations.skforecast import SkforecastOptCV
from hyperactive.opt import HillClimbing

try:
    from skforecast.recursive import ForecasterRecursive
except ImportError:
    pass


@pytest.fixture
def data():
    """Create test data."""
    return pd.Series(
        np.random.randn(100),
        index=pd.date_range(start="2020-01-01", periods=100, freq="D"),
        name="y",
    )


def test_skforecast_opt_cv(data):
    """Test SkforecastOptCV."""
    pytest.importorskip("skforecast")

    forecaster = ForecasterRecursive(
        regressor=RandomForestRegressor(random_state=123), lags=5
    )

    optimizer = HillClimbing(
        search_space={
            "n_estimators": [10, 20],
            "max_depth": [2, 5],
        },
        n_iter=2,
    )

    opt_cv = SkforecastOptCV(
        forecaster=forecaster,
        optimizer=optimizer,
        steps=5,
        metric="mean_squared_error",
        initial_train_size=50,
        verbose=False,
    )

    opt_cv.fit(y=data)
    predictions = opt_cv.predict(steps=5)

    assert len(predictions) == 5
    assert isinstance(predictions, pd.Series)
    assert "n_estimators" in opt_cv.best_params_
    assert "max_depth" in opt_cv.best_params_
