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
    """Simulate SVM optimization with kernel-specific parameters.

    The objective simulates finding optimal SVM hyperparameters.
    - Best C is around 1.0 (log10(C) = 0)
    - For RBF kernel: best gamma is around 0.1 (log10(gamma) = -1)
    - For poly kernel: best degree is 3
    """
    kernel = params["kernel"]
    C = params["C"]

    # Base score from C (optimal at C=1.0)
    score = -np.log10(C) ** 2

    # Kernel-specific scoring
    if kernel == "rbf":
        gamma = params.get("gamma", 1.0)
        # Optimal at gamma=0.1
        score -= (np.log10(gamma) + 1) ** 2
    elif kernel == "poly":
        degree = params.get("degree", 3)
        # Optimal at degree=3
        score -= (degree - 3) ** 2
    # linear kernel has no additional parameters

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

    # Run optimization
    print("\n--- Running Optimization ---")

    optimizer = RandomSearch(
        search_space=space,
        n_iter=50,
        experiment=svm_objective,
    )

    result = optimizer.solve()

    print(f"\nBest parameters found:")
    print(f"  kernel: {result['kernel']}")
    print(f"  C: {result['C']:.4f}")

    # Show kernel-specific parameters
    active_result = space.filter_active_params(result)
    if "gamma" in active_result:
        print(f"  gamma: {result['gamma']:.6f}")
    if "degree" in active_result:
        print(f"  degree: {result['degree']}")

    print(f"  Score: {svm_objective(result):.4f}")

    # Method chaining for fluent API
    print("\n--- Neural Network Example with Chained Conditions ---")

    nn_space = (
        SearchSpace(
            activation=["relu", "tanh", "sigmoid"],
            hidden_size=np.arange(32, 513, 32),
            dropout=np.arange(0.0, 0.6, 0.1),
            alpha=(1e-5, 1e-2, "log"),  # L2 regularization
        )
        .add_condition("dropout", when=lambda p: p["activation"] == "relu")
        .add_condition("alpha", when=lambda p: p["hidden_size"] > 128)
    )

    def nn_objective(params):
        """Simulate neural network hyperparameter optimization."""
        hidden = params["hidden_size"]
        activation = params["activation"]

        # Base score from hidden size (optimal around 256)
        score = -(((hidden - 256) / 100) ** 2)

        # Activation bonus
        if activation == "relu":
            score += 0.5
            # Dropout penalty (optimal around 0.2 for relu)
            dropout = params.get("dropout", 0.0)
            score -= (dropout - 0.2) ** 2
        elif activation == "tanh":
            score += 0.3

        # Regularization (only matters for large models)
        if hidden > 128:
            alpha = params.get("alpha", 1e-3)
            score -= (np.log10(alpha) + 3) ** 2 * 0.1

        return score

    optimizer = RandomSearch(
        search_space=nn_space,
        n_iter=50,
        experiment=nn_objective,
    )

    result = optimizer.solve()

    print(f"\nBest neural network parameters:")
    print(f"  activation: {result['activation']}")
    print(f"  hidden_size: {result['hidden_size']}")

    active_result = nn_space.filter_active_params(result)
    if "dropout" in active_result:
        print(f"  dropout: {result['dropout']:.2f}")
    if "alpha" in active_result:
        print(f"  alpha: {result['alpha']:.6f}")

    print(f"  Score: {nn_objective(result):.4f}")


if __name__ == "__main__":
    main()
