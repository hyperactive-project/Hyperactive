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
    n_iter=100,
    experiment=objective,
)
best_params = optimizer.solve()

print(f"Best parameters: {best_params}")
# [end:full_example]

if __name__ == "__main__":
    # Verify the optimization found parameters close to (0, 0)
    assert abs(best_params["x"]) < 1.0, f"Expected x near 0, got {best_params['x']}"
    assert abs(best_params["y"]) < 1.0, f"Expected y near 0, got {best_params['y']}"
    print("Quick start example passed!")
