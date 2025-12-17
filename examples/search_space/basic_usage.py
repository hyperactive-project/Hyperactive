"""Basic SearchSpace usage with automatic type inference.

This example demonstrates the core SearchSpace functionality:
- Automatic dimension type inference from Python types
- Log-scale specification for learning rates
- Using SearchSpace with different optimizer backends
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace
from hyperactive.opt import HillClimbing, RandomSearch


def sphere_function(params):
    """Simple sphere function (minimize x^2 + y^2)."""
    return -(params["x"] ** 2 + params["y"] ** 2)


def main():
    # SearchSpace with automatic type inference
    # - list → categorical
    # - tuple (low, high) → continuous
    # - tuple (low, high, "log") → log-scale continuous
    # - numpy array → discrete
    # - scalar → constant

    space = SearchSpace(
        x=np.arange(-5, 5, 0.1),  # discrete: numpy array
        y=np.arange(-5, 5, 0.1),  # discrete: numpy array
    )

    print("SearchSpace dimensions:")
    for name, dim in space.dimensions.items():
        print(f"  {name}: {dim.dim_type.value}")

    # Use with HillClimbing optimizer
    print("\nOptimizing with HillClimbing...")
    optimizer = HillClimbing(
        search_space=space,
        n_iter=50,
        experiment=sphere_function,
    )
    result = optimizer.solve()
    print(f"Best result: x={result['x']:.2f}, y={result['y']:.2f}")
    print(f"Best score: {sphere_function(result):.4f}")

    # More complex search space with different types
    complex_space = SearchSpace(
        learning_rate=(1e-5, 1e-1, "log"),  # log-scale continuous
        batch_size=[16, 32, 64, 128, 256],  # categorical (list)
        optimizer=["adam", "sgd", "rmsprop"],  # categorical (strings)
        epochs=(10, 100),  # continuous integer range
        seed=42,  # constant
    )

    print("\nComplex SearchSpace:")
    for name, dim in complex_space.dimensions.items():
        print(f"  {name}: {dim.dim_type.value}")

    # SearchSpace can also be created from a dict
    dict_space = SearchSpace(
        {
            "x": np.linspace(-10, 10, 100),
            "y": np.linspace(-10, 10, 100),
        }
    )

    print("\nDict-based SearchSpace:")
    print(f"  Parameters: {list(dict_space.param_names)}")


if __name__ == "__main__":
    main()
