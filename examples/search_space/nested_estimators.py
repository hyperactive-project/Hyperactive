"""Nested search spaces for estimator selection.

This example demonstrates how to use nested search spaces
for selecting between different estimators, each with their
own hyperparameters. This is common in AutoML scenarios.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace


def main():
    # Nested search space using the _params suffix convention
    # The keys of the dict become the categorical choices for "estimator"
    # and each value defines the hyperparameters for that estimator

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
    except ImportError:
        print("This example requires scikit-learn")
        return

    space = SearchSpace(
        estimator_params={
            RandomForestClassifier: {
                "n_estimators": np.arange(10, 201, 10),
                "max_depth": [3, 5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
            },
            SVC: {
                "C": (0.01, 100.0, "log"),
                "kernel": ["rbf", "linear"],
                "gamma": (1e-4, 10.0, "log"),
            },
            GradientBoostingClassifier: {
                "n_estimators": np.arange(50, 301, 50),
                "learning_rate": (0.01, 0.3, "log"),
                "max_depth": [3, 5, 7],
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
    for cond in space.conditions:
        print(f"  - {cond.target_param} depends on {cond.depends_on}")

    # Union operator to combine search spaces
    print("\n--- Union Operator Example ---")

    base_space = SearchSpace(
        learning_rate=(1e-5, 1e-1, "log"),
        batch_size=[32, 64, 128],
    )

    augmentation_space = SearchSpace(
        augment=[True, False],
        flip_probability=(0.0, 1.0),
    )

    # Combine with | operator (union)
    combined = base_space | augmentation_space

    print("Combined SearchSpace:")
    print(f"  Parameters: {list(combined.param_names)}")

    # Union with conflict resolution
    space_v1 = SearchSpace(lr=(1e-4, 1e-2))
    space_v2 = SearchSpace(lr=(1e-5, 1e-1))  # different range

    # Default: last wins
    merged = space_v1 | space_v2
    print(f"\nUnion (last wins): lr bounds = "
          f"({merged.dimensions['lr'].low}, {merged.dimensions['lr'].high})")

    # First wins
    merged = space_v1.union(space_v2, on_conflict="first")
    print(f"Union (first wins): lr bounds = "
          f"({merged.dimensions['lr'].low}, {merged.dimensions['lr'].high})")


if __name__ == "__main__":
    main()
