"""Advanced examples for the examples.rst page.

This snippet file contains runnable examples demonstrating Hyperactive's
advanced functionality like warm starting and optimizer comparison.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import HillClimbing

# Setup common fixtures for examples
X, y = load_wine(return_X_y=True)
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    X=X, y=y, cv=3,
)
search_space = {
    "n_estimators": list(range(10, 101, 10)),
    "max_depth": list(range(1, 11)),
    "min_samples_split": list(range(2, 11)),
}


# [start:warm_starting]
from hyperactive.opt.gfo import HillClimbing

# Previous best parameters
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
]

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=40,
    experiment=experiment,
    initialize={"warm_start": warm_start_points},
)
best_params = optimizer.solve()
# [end:warm_starting]


# [start:comparing_optimizers]
from hyperactive.opt.gfo import (
    HillClimbing,
    RandomSearch,
    BayesianOptimizer,
    ParticleSwarmOptimizer,
)

optimizers = {
    "HillClimbing": HillClimbing,
    "RandomSearch": RandomSearch,
    "Bayesian": BayesianOptimizer,
    "ParticleSwarm": ParticleSwarmOptimizer,
}

results = {}
for name, OptClass in optimizers.items():
    optimizer = OptClass(
        search_space=search_space,
        n_iter=50,
        experiment=experiment,
        random_state=42,
    )
    best = optimizer.solve()
    score, _ = experiment.score(best)
    results[name] = {"params": best, "score": score}
    print(f"{name}: score={score:.4f}")
# [end:comparing_optimizers]


if __name__ == "__main__":
    print("Advanced examples passed!")
    print(f"Best optimizer results: {results}")
