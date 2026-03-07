"""Bayesian Optimizer example for documentation.

This snippet demonstrates the usage of BayesianOptimizer for
optimization problems. It is included in get_started.rst.
"""

# [start:full_example]
# [end:full_example]
# Need to define experiment and search_space for standalone execution
import numpy as np

from hyperactive.opt.gfo import BayesianOptimizer


def experiment(params):
    """Compute simple objective function."""
    x = params["x"]
    y = params["y"]
    return -(x**2 + y**2)


search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

# [start:optimizer_usage]
optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=experiment,
)
best_params = optimizer.solve()
# [end:optimizer_usage]

if __name__ == "__main__":
    print(f"Best parameters: {best_params}")
    # Verify the optimization returned valid parameters
    assert "x" in best_params and "y" in best_params
    assert -5 <= best_params["x"] <= 5, f"x out of range: {best_params['x']}"
    assert -5 <= best_params["y"] <= 5, f"y out of range: {best_params['y']}"
    print("Bayesian optimizer example passed!")
