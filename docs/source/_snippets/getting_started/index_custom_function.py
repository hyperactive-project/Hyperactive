"""Custom function example for index page.

This snippet demonstrates the basic custom function optimization
shown on the landing page. It is included in index.rst.
"""

# [start:full_example]
import numpy as np
from hyperactive.opt.gfo import HillClimbing


# Define your objective function
def objective(params):
    x, y = params["x"], params["y"]
    return -(x**2 + y**2)  # Maximize (minimize negative)


# Define the search space
search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

# Create optimizer and solve
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
    print("Index custom function example passed!")
