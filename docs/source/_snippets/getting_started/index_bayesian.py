"""Bayesian optimization example for index page.

This snippet demonstrates Bayesian optimization with a more complex
objective function shown on the landing page. It is included in index.rst.
"""

# [start:full_example]
import numpy as np
from hyperactive.opt.gfo import BayesianOptimizer


def complex_objective(params):
    x = params["x"]
    y = params["y"]
    return -((x - 2) ** 2 + (y + 1) ** 2) + np.sin(x * y)


search_space = {
    "x": np.linspace(-5, 5, 100),
    "y": np.linspace(-5, 5, 100),
}

optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=50,
    experiment=complex_objective,
)
best_params = optimizer.solve()
# [end:full_example]

if __name__ == "__main__":
    print(f"Best parameters: {best_params}")
    # Verify we got valid parameters
    assert "x" in best_params
    assert "y" in best_params
    print("Index Bayesian example passed!")
