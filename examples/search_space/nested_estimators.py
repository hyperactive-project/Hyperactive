"""Nested search spaces for estimator selection.

This example demonstrates how to use nested search spaces
for selecting between different estimators, each with their
own hyperparameters. This is common in AutoML scenarios.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace
from hyperactive.opt import RandomSearch


def main():
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC
    except ImportError:
        print("This example requires scikit-learn")
        return

    # Generate a sample dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    # Nested search space: dict with class keys is automatically detected
    # The keys become categorical choices for "estimator"
    # and each value defines the hyperparameters for that estimator
    space = SearchSpace(
        estimator={
            RandomForestClassifier: {
                "n_estimators": np.arange(10, 101, 10),
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
            SVC: {
                "C": (0.1, 10.0, "log"),
                "kernel": ["rbf", "linear"],
            },
            GradientBoostingClassifier: {
                "n_estimators": np.arange(20, 101, 20),
                "learning_rate": (0.01, 0.3, "log"),
                "max_depth": [3, 5],
            },
        },
    )

    print("Nested SearchSpace for estimator selection:")
    print(f"  Total dimensions: {len(space)}")
    print(f"  Has nested spaces: {space.has_nested_spaces}")
    print(f"  Has conditions: {space.has_conditions}")

    print("\nAll parameters:")
    for name, dim in space.dimensions.items():
        print(f"  {name}: {dim.dim_type.value}")

    print(f"\nAutomatic conditions created: {len(space.conditions)}")

    # Define objective - nested params access works automatically!
    def estimator_objective(params):
        """Evaluate estimator with cross-validation.

        With nested search spaces, params automatically supports:
        - params["estimator"]["n_estimators"] -> access individual param
        - params["estimator"]() -> auto-instantiate with all params
        - params["estimator"](n_jobs=-1) -> instantiate with override
        - params["estimator"] == SomeClass -> comparison works
        """
        # One-liner model creation with all optimized parameters!
        estimator = params["estimator"]()

        # Evaluate with cross-validation
        scores = cross_val_score(estimator, X, y, cv=3, scoring="accuracy")

        return scores.mean()

    # Run optimization
    print("\n--- Running Optimization ---")

    optimizer = RandomSearch(
        search_space=space,
        n_iter=30,
        experiment=estimator_objective,
    )

    result = optimizer.solve()

    # Wrap result for clean access
    best = space.wrap_params(result)

    print(f"\nBest estimator: {best['estimator'].value.__name__}")
    print("Best parameters:")
    for param_name, value in best["estimator"].items():
        print(f"  {param_name}: {value}")

    print(f"Best CV accuracy: {estimator_objective(result):.4f}")

    # Union operator example with actual optimization
    print("\n--- Union Operator Example ---")

    base_space = SearchSpace(
        learning_rate=(1e-4, 1e-1, "log"),
        batch_size=[32, 64, 128],
    )

    regularization_space = SearchSpace(
        weight_decay=(1e-6, 1e-2, "log"),
        dropout=np.arange(0.0, 0.6, 0.1),
    )

    # Combine with | operator
    combined = base_space | regularization_space

    print("Combined SearchSpace:")
    print(f"  Parameters: {list(combined.param_names)}")

    def training_objective(params):
        """Simulate training optimization."""
        lr = params["learning_rate"]
        batch = params["batch_size"]
        wd = params["weight_decay"]
        dropout = params["dropout"]

        # Optimal: lr=0.01, batch=64, wd=1e-4, dropout=0.2
        score = 0
        score -= (np.log10(lr) + 2) ** 2
        score -= ((batch - 64) / 32) ** 2
        score -= (np.log10(wd) + 4) ** 2
        score -= (dropout - 0.2) ** 2

        return score

    optimizer = RandomSearch(
        search_space=combined,
        n_iter=50,
        experiment=training_objective,
    )

    result = optimizer.solve()

    print(f"\nBest training parameters:")
    print(f"  learning_rate: {result['learning_rate']:.6f}")
    print(f"  batch_size: {result['batch_size']}")
    print(f"  weight_decay: {result['weight_decay']:.8f}")
    print(f"  dropout: {result['dropout']:.2f}")
    print(f"  Score: {training_objective(result):.4f}")


if __name__ == "__main__":
    main()
