"""
Skforecast Integration Example - Hyperparameter Tuning for Time Series Forecasting

This example demonstrates how to use Hyperactive to tune hyperparameters of a
skforecast ForecasterRecursive model. It uses the SkforecastOptCV class which
provides a familiar sklearn-like API for integrating skforecast models with
Hyperactive's optimization algorithms.

Characteristics:
- Integration with skforecast's backtesting functionality
- Tuning of regressor hyperparameters (e.g., RandomForestRegressor)
- Uses HillClimbing optimizer (can be swapped for any Hyperactive optimizer)
- Time series cross-validation via backtesting
"""

import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import RandomForestRegressor
from hyperactive.opt import HillClimbing
from hyperactive.integrations.skforecast import SkforecastOptCV

# Generate synthetic data
data = pd.Series(
    np.random.randn(100),
    index=pd.date_range(start="2020-01-01", periods=100, freq="D"),
    name="y",
)

# Define forecaster
forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123), lags=5
)

# Define optimizer
optimizer = HillClimbing(
    search_space={
        "n_estimators": list(range(10, 100, 10)),
        "max_depth": list(range(2, 10)),
    },
    n_iter=10,
)

# Define SkforecastOptCV
opt_cv = SkforecastOptCV(
    forecaster=forecaster,
    optimizer=optimizer,
    steps=5,
    metric="mean_squared_error",
    initial_train_size=50,
    verbose=True,
)

# Fit
print("Fitting...")
opt_cv.fit(y=data)

# Predict
print("Predicting...")
predictions = opt_cv.predict(steps=5)
print("Predictions:")
print(predictions)
print("Best params:", opt_cv.best_params_)
