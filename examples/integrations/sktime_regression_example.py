import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sktime.datasets import load_unit_test
from sktime.transformations.panel.rocket import Rocket

from hyperactive.integrations.sktime import TSROptCV
from hyperactive.opt import RandomSearch

# 1. Load data
X_train, y_train = load_unit_test(split="train", return_X_y=True)
X_test, y_test = load_unit_test(split="test", return_X_y=True)

# 2. Define search space
# We use a pipeline with Rocket transform and DecisionTreeRegressor
# But TSROptCV wraps a regressor.
# Let's use a simple regressor that handles time series or use a pipeline.
# For simplicity in this example, we can use a ComposableTimeSeriesForestRegressor if available,
# or just wrap a sklearn regressor if we treat it as a tabular problem (which sktime can do).
# However, TSROptCV expects a sktime regressor.

from sktime.regression.dummy import DummyRegressor
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

# Let's use KNeighborsTimeSeriesRegressor as it is a standard sktime regressor
search_space_knn = {
    "n_neighbors": list(range(1, 10)),
    "weights": ["uniform", "distance"],
}

tsr_opt = TSROptCV(
    estimator=KNeighborsTimeSeriesRegressor(),
    optimizer=RandomSearch(search_space_knn, n_iter=5),
    cv=3,
)

# 4. Run optimization
tsr_opt.fit(X_train, y_train)

# 5. Check results
print("Best score:", tsr_opt.best_score_)
print("Best params:", tsr_opt.best_params_)

# 6. Predict
y_pred = tsr_opt.predict(X_test)
print("Predictions shape:", y_pred.shape)
