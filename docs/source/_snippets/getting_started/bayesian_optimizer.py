"""Bayesian Optimizer example for documentation.

This snippet demonstrates the usage of BayesianOptimizer for
optimization problems. It is included in get_started.rst.
"""

# [start:full_example]
from hyperactive.opt.gfo import BayesianOptimizer
# [end:full_example]

# Need to define experiment and search_space for standalone execution
import numpy as np


def experiment(params):
    """Simple objective function."""
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
    n_iter=30,
    experiment=experiment,
)
best_params = optimizer.solve()
# [end:optimizer_usage]

if __name__ == "__main__":
    print(f"Best parameters: {best_params}")
    # Verify the optimization found parameters close to (0, 0)
    assert abs(best_params["x"]) < 2.0, f"Expected x near 0, got {best_params['x']}"
    assert abs(best_params["y"]) < 2.0, f"Expected y near 0, got {best_params['y']}"
    print("Bayesian optimizer example passed!")
