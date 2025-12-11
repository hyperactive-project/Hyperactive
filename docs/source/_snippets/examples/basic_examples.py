"""Basic examples for the examples.rst page.

This snippet file contains runnable examples demonstrating Hyperactive's
basic functionality including custom function and sklearn optimization.
"""

import numpy as np

# [start:custom_function]
import numpy as np
from hyperactive.opt.gfo import HillClimbing


def objective(params):
    x = params["x"]
    y = params["y"]
    return -(x**2 + y**2)  # Maximize (minimize the parabola)


search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=objective,
)
best_params = optimizer.solve()
print(f"Best parameters: {best_params}")
# [end:custom_function]


# [start:sklearn_tuning]
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import HillClimbing

# Load data
X, y = load_wine(return_X_y=True)

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    X=X, y=y, cv=3,
)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201)),
    "max_depth": list(range(1, 21)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 11)),
}

# Optimize
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    experiment=experiment,
)
best_params = optimizer.solve()
# [end:sklearn_tuning]


if __name__ == "__main__":
    print("Basic examples passed!")
    print(f"Custom function best: {best_params}")
