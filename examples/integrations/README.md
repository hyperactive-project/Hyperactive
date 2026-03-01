# Integrations with AI framework toolboxes

This directory contains examples for estimator level integration with
common AI toolbox libraries such as `scikit-learn` or `sktime`.

## Quick Start

For full code, see below.

You can also run any example directly:
```bash
python sklearn_classif_example.py
python sktime_forecasting_example.py
python sktime_tsc_example.py
```

Requires `scikit-learn` resp `sktime` installed.

## Available Integrations

| Integration | Class name |
|-----------|-------------------|
| `sklearn` classifier or regressor tuner | [OptCV](sklearn_classif_example.py) |
| `sktime` forecasting tuner | [ForecastingOptCV](sktime_forecasting_example.py) |
| `sktime` time series classifier tuner | [TSCOptCV](sktime_tsc_example.py) |

## Integration with sklearn

Any available tuning engine from hyperactive can be used, for example:

* grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
* hill climbing - ``from hyperactive.opt import HillClimbing``
* optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

```python
# For illustration, we use grid search, this can be replaced by any other optimizer.

# 1. defining the tuned estimator:
from sklearn.svm import SVC
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"kernel": ["linear", "rbf"], "C": [1, 10]}
tuned_svc = OptCV(SVC(), GridSearch(param_grid))

# 2. fitting the tuned estimator:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tuned_svc.fit(X_train, y_train)

# 3. obtaining predictions
y_pred = tuned_svc.predict(X_test)

# 4. obtaining best parameters and best estimator
best_params = tuned_svc.best_params_
best_estimator = tuned_svc.best_estimator_
```

## Integration with sktime - forecasting

Any available tuning engine from hyperactive can be used, for example:

* grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
* hill climbing - ``from hyperactive.opt import HillClimbing``
* optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

For illustration, we use grid search, this can be replaced by any other optimizer.

```python
# 1. defining the tuned estimator:
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from hyperactive.integrations.sktime import ForecastingOptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"strategy": ["mean", "last", "drift"]}
tuned_naive = ForecastingOptCV(
    NaiveForecaster(),
    GridSearch(param_grid),
    cv=ExpandingWindowSplitter(
        initial_window=12, step_length=3, fh=range(1, 13)
    ),
)

# 2. fitting the tuned estimator:
from sktime.datasets import load_airline
from sktime.split import temporal_train_test_split
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

tuned_naive.fit(y_train, fh=range(1, 13))

# 3. obtaining predictions
y_pred = tuned_naive.predict()

# 4. obtaining best parameters and best forecaster
best_params = tuned_naive.best_params_
best_forecaster = tuned_naive.best_forecaster_
```

## Integration with sktime - time series classification

Any available tuning engine from hyperactive can be used, for example:

* grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
* hill climbing - ``from hyperactive.opt import HillClimbing``
* optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

For illustration, we use grid search, this can be replaced by any other optimizer.

```python
# 1. defining the tuned estimator:
from sktime.classification.dummy import DummyClassifier
from sklearn.model_selection import KFold
from hyperactive.integrations.sktime import TSCOptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"strategy": ["most_frequent", "stratified"]}
tuned_naive = TSCOptCV(
    DummyClassifier(),
    GridSearch(param_grid),
    cv=KFold(n_splits=2, shuffle=False),
)

# 2. fitting the tuned estimator:
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(
    return_X_y=True, split="TRAIN", return_type="pd-multiindex"
)
X_test, _ = load_unit_test(
    return_X_y=True, split="TEST", return_type="pd-multiindex"
)

tuned_naive.fit(X_train, y_train)

# 3. obtaining predictions
y_pred = tuned_naive.predict(X_test)

# 4. obtaining best parameters and best estimator
best_params = tuned_naive.best_params_
best_classifier = tuned_naive.best_estimator_
```
