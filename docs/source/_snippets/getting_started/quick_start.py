"""Quick start example for documentation.

This snippet demonstrates the basic usage of Hyperactive for optimizing
a custom objective function. It is included in get_started.rst.
"""

# [start:full_example]
import numpy as np
from hyperactive.opt.gfo import HillClimbing


# 1. Define your objective function
def objective(params):
    x = params["x"]
    y = params["y"]
    return -(x**2 + y**2)  # Hyperactive maximizes by default


# 2. Define the search space
search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

# 3. Create an optimizer and solve
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
best_params = optimizer.solve()

print(f"Best parameters: {best_params}")
# [end:full_example]

if __name__ == "__main__":
    # Verify the optimization returned valid parameters
    assert "x" in best_params and "y" in best_params
    assert -5 <= best_params["x"] <= 5, f"x out of range: {best_params['x']}"
    assert -5 <= best_params["y"] <= 5, f"y out of range: {best_params['y']}"
    print("Quick start example passed!")
