"""Introduction page code snippets for documentation.

This snippet file contains examples from the introduction.rst page.
Some snippets are illustrative (showing patterns) while others are runnable.
"""

import numpy as np

# Define placeholders for illustrative code
# These allow the file to be imported without errors
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])
X = X_train
y = y_train

# [start:simple_objective]
def objective(params):
    x = params["x"]
    y = params["y"]
    # Return a score to maximize
    return -(x**2 + y**2)
# [end:simple_objective]


# [start:sklearn_experiment_intro]
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    cv=5,
)
# [end:sklearn_experiment_intro]


# [start:optimizer_imports]
from hyperactive.opt.gfo import (
    HillClimbing,           # Local search
    RandomSearch,           # Global search
    BayesianOptimizer,      # Sequential model-based
    ParticleSwarmOptimizer, # Population-based
)
# [end:optimizer_imports]


# [start:search_space_definition]
import numpy as np

search_space = {
    # Discrete integer values
    "n_estimators": list(range(10, 200, 10)),

    # Continuous values (discretized)
    "learning_rate": np.logspace(-4, 0, 20),

    # Categorical values
    "kernel": ["linear", "rbf", "poly"],
}
# [end:search_space_definition]


# [start:workflow_experiment_options]
# Option A: Custom function
def my_objective(params):
    # Your evaluation logic here
    return score

# Option B: Built-in sklearn experiment
from hyperactive.experiment.integrations import SklearnCvExperiment

experiment = SklearnCvExperiment(
    estimator=YourEstimator(),
    X=X, y=y, cv=5,
)
# [end:workflow_experiment_options]


# [start:workflow_search_space]
search_space = {
    "param1": [1, 2, 3, 4, 5],
    "param2": np.linspace(0.1, 1.0, 10),
    "param3": ["option_a", "option_b"],
}
# [end:workflow_search_space]


# [start:workflow_optimizer]
from hyperactive.opt.gfo import HillClimbing

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,           # Number of iterations
    experiment=experiment,
    random_state=42,      # For reproducibility
)
# [end:workflow_optimizer]


# [start:workflow_solve]
best_params = optimizer.solve()
print(f"Best parameters: {best_params}")
# [end:workflow_solve]


# [start:warm_starting]
warm_start = [
    {"n_estimators": 100, "max_depth": 10},  # Start from known good point
]

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=50,
    experiment=experiment,
    initialize={"warm_start": warm_start},
)
# [end:warm_starting]


if __name__ == "__main__":
    # The actual test code runs here
    print("Introduction snippet file is importable!")
    print("Full integration test runs in test_doc_snippets.py")
