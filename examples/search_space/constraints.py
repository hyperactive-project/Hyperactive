"""Constraints in SearchSpace.

This example demonstrates how to add constraints to filter
out invalid parameter combinations. Constraints are useful for:
- Memory/compute budget limits
- Parameter interdependencies
- Domain-specific validity rules
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace
from hyperactive.opt import HillClimbing


def constrained_objective(params):
    """Objective with implicit constraint awareness."""
    batch_size = params["batch_size"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]

    # Score based on model capacity
    capacity = hidden_size * num_layers
    return capacity / 1000 - (batch_size / 256) ** 2


def main():
    # Search space with memory constraints
    space = SearchSpace(
        batch_size=[32, 64, 128, 256, 512],
        hidden_size=[128, 256, 512, 1024],
        num_layers=[2, 4, 6, 8],
    )

    # Memory constraint: batch_size * hidden_size * num_layers < threshold
    space.add_constraint(
        lambda p: p["batch_size"] * p["hidden_size"] * p["num_layers"] < 500000,
        name="memory_limit",
    )

    # Additional constraint: large batches need smaller models
    space.add_constraint(
        lambda p: not (p["batch_size"] >= 256 and p["hidden_size"] >= 512),
        name="large_batch_small_model",
    )

    print("Constrained SearchSpace:")
    print(f"  Parameters: {list(space.param_names)}")
    print(f"  Has constraints: {space.has_constraints}")
    print(f"  Number of constraints: {len(space.constraints)}")

    # Test constraint checking
    print("\nConstraint checking examples:")

    valid_params = {"batch_size": 64, "hidden_size": 256, "num_layers": 4}
    print(f"  {valid_params}")
    print(f"    Valid: {space.check_constraints(valid_params)}")

    invalid_params = {"batch_size": 512, "hidden_size": 1024, "num_layers": 8}
    print(f"  {invalid_params}")
    print(f"    Valid: {space.check_constraints(invalid_params)}")

    borderline_params = {"batch_size": 256, "hidden_size": 512, "num_layers": 2}
    print(f"  {borderline_params}")
    print(f"    Valid: {space.check_constraints(borderline_params)}")

    # Method chaining with conditions and constraints
    print("\n--- Combined Conditions and Constraints ---")

    ml_space = (
        SearchSpace(
            model=["linear", "tree", "neural"],
            learning_rate=(1e-5, 1e-1, "log"),
            n_estimators=[10, 50, 100, 200],
            hidden_layers=[1, 2, 3, 4],
            regularization=(1e-6, 1e-2, "log"),
        )
        # Conditions: model-specific parameters
        .add_condition("n_estimators", when=lambda p: p["model"] == "tree")
        .add_condition("hidden_layers", when=lambda p: p["model"] == "neural")
        .add_condition("learning_rate", when=lambda p: p["model"] != "linear")
        # Constraints: valid combinations
        .add_constraint(
            lambda p: not (p["model"] == "neural" and p.get("hidden_layers", 1) > 3
                          and p.get("learning_rate", 0.01) > 0.01),
            name="stable_deep_training",
        )
    )

    print(f"ML SearchSpace:")
    print(f"  Parameters: {list(ml_space.param_names)}")
    print(f"  Conditions: {len(ml_space.conditions)}")
    print(f"  Constraints: {len(ml_space.constraints)}")

    # Using with optimizer (GFO backend supports constraints natively)
    print("\n--- Using with Optimizer ---")

    simple_space = SearchSpace(
        x=np.arange(0, 10, 0.5),
        y=np.arange(0, 10, 0.5),
    )
    simple_space.add_constraint(lambda p: p["x"] + p["y"] < 10)

    def objective(params):
        return params["x"] + params["y"]

    optimizer = HillClimbing(
        search_space=simple_space,
        n_iter=50,
        experiment=objective,
    )

    result = optimizer.solve()
    print(f"Best result: x={result['x']:.1f}, y={result['y']:.1f}")
    print(f"Sum: {result['x'] + result['y']:.1f} (constraint: < 10)")
    print(f"Constraint satisfied: {result['x'] + result['y'] < 10}")


if __name__ == "__main__":
    main()
