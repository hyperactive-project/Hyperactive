"""Conditional dimensions with SearchSpace.

This example demonstrates how to define parameters that are only
active under certain conditions, which is useful for:
- Algorithm-specific hyperparameters
- Kernel-specific parameters in SVM
- Architecture-specific options in neural networks
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace
from hyperactive.opt import RandomSearch


def svm_objective(params):
    """Simulate SVM optimization with kernel-specific parameters."""
    kernel = params["kernel"]
    C = params["C"]

    # Base score from C
    score = -np.log10(C) ** 2

    # Kernel-specific scoring
    if kernel == "rbf":
        gamma = params.get("gamma", 1.0)
        score -= np.log10(gamma) ** 2
    elif kernel == "poly":
        degree = params.get("degree", 3)
        score -= (degree - 3) ** 2

    return score


def main():
    # SVM-style search space with conditional parameters
    space = SearchSpace(
        kernel=["rbf", "linear", "poly"],
        C=(0.01, 100.0, "log"),
        gamma=(1e-4, 10.0, "log"),  # only for rbf kernel
        degree=[2, 3, 4, 5],  # only for poly kernel
    )

    # Add conditions: gamma is only used with rbf kernel
    space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

    # degree is only used with poly kernel
    space.add_condition("degree", when=lambda p: p["kernel"] == "poly")

    print("SVM SearchSpace:")
    print(f"  Parameters: {list(space.param_names)}")
    print(f"  Has conditions: {space.has_conditions}")
    print(f"  Number of conditions: {len(space.conditions)}")

    # Test parameter filtering
    print("\nParameter filtering examples:")

    rbf_params = {"kernel": "rbf", "C": 1.0, "gamma": 0.1, "degree": 3}
    active = space.filter_active_params(rbf_params)
    print(f"  RBF kernel active params: {list(active.keys())}")

    linear_params = {"kernel": "linear", "C": 1.0, "gamma": 0.1, "degree": 3}
    active = space.filter_active_params(linear_params)
    print(f"  Linear kernel active params: {list(active.keys())}")

    poly_params = {"kernel": "poly", "C": 1.0, "gamma": 0.1, "degree": 3}
    active = space.filter_active_params(poly_params)
    print(f"  Poly kernel active params: {list(active.keys())}")

    # Method chaining for fluent API
    space2 = (
        SearchSpace(
            activation=["relu", "tanh", "sigmoid"],
            hidden_size=(32, 512),
            dropout=(0.0, 0.5),
            alpha=(1e-5, 1e-2, "log"),  # L2 regularization
        )
        .add_condition("dropout", when=lambda p: p["activation"] == "relu")
        .add_condition("alpha", when=lambda p: p["hidden_size"] > 128)
    )

    print("\nNeural network SearchSpace (chained):")
    print(f"  Parameters: {list(space2.param_names)}")
    print(f"  Conditions: {len(space2.conditions)}")


if __name__ == "__main__":
    main()
