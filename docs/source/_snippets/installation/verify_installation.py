"""Installation verification snippet.

This snippet demonstrates how to verify Hyperactive installation.
"""

# [start:verify_installation]
import hyperactive
print(f"Hyperactive version: {hyperactive.__version__}")

# Quick test
import numpy as np
from hyperactive.opt.gfo import HillClimbing


def objective(params):
    return -(params["x"] ** 2)


optimizer = HillClimbing(
    search_space={"x": np.arange(-5, 5, 0.1)},
    n_iter=10,
    experiment=objective,
)
best = optimizer.solve()
print(f"Test optimization successful: {best}")
# [end:verify_installation]


if __name__ == "__main__":
    print("Installation verification passed!")
