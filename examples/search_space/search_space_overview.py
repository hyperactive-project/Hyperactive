"""SearchSpace creation overview.

This example demonstrates how to create search spaces using the unified
SearchSpace class. It covers all major features:

- Dimension types (categorical, continuous, discrete, log-scale, constant)
- Conditions (conditional parameter activation)
- Constraints (filtering invalid parameter combinations)
- Nested search spaces (hierarchical parameter structures)
- Union of search spaces (combining spaces)
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive import SearchSpace

# =============================================================================
# Dimension Types
# =============================================================================

# SearchSpace infers dimension types from Python types:
# - list          -> categorical
# - tuple (int)   -> continuous integer
# - tuple (float) -> continuous float
# - tuple + "log" -> log-scale continuous
# - numpy array   -> discrete
# - scalar        -> constant

dimension_types = SearchSpace(
    # Categorical: list of choices (strings, numbers, classes, functions, ...)
    kernel=["rbf", "linear", "poly"],
    activation=["relu", "tanh", "sigmoid", "gelu"],
    # Continuous integer: tuple of two integers
    n_layers=(1, 10),
    batch_size=(16, 256),
    # Continuous float: tuple with at least one float
    dropout=(0.0, 0.5),
    weight_decay=(0.0, 0.1),
    # Log-scale continuous: tuple with "log" as third element
    learning_rate=(1e-5, 1e-1, "log"),
    C=(0.01, 100.0, "log"),
    # Discrete: numpy array of specific values
    hidden_size=np.array([32, 64, 128, 256, 512]),
    x_grid=np.arange(-5.0, 5.0, 0.1),
    # Constant: single value (not searched)
    random_state=42,
    verbose=False,
)

print(f"Dimension types example: {len(dimension_types)} parameters")


# =============================================================================
# Conditions
# =============================================================================

# Conditions control when a parameter is active based on other parameter values.
# Use conditions when a parameter only makes sense for certain configurations.

svm_space = SearchSpace(
    kernel=["rbf", "linear", "poly"],
    C=(0.01, 100.0, "log"),
    gamma=(1e-4, 10.0, "log"),
    degree=[2, 3, 4, 5],
)

# gamma is only relevant for rbf kernel
svm_space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

# degree is only relevant for poly kernel
svm_space.add_condition("degree", when=lambda p: p["kernel"] == "poly")

print(f"SVM space: {len(svm_space.conditions)} conditions")


# Method chaining with fluent API
nn_space = (
    SearchSpace(
        activation=["relu", "tanh", "sigmoid"],
        hidden_size=(32, 512),
        dropout=(0.0, 0.5),
        batch_norm=[True, False],
    )
    .add_condition("dropout", when=lambda p: p["activation"] == "relu")
    .add_condition("batch_norm", when=lambda p: p["hidden_size"] > 128)
)


# =============================================================================
# Constraints
# =============================================================================

# Constraints filter out invalid parameter combinations.
# Use constraints when parameters interact in ways that make some combinations invalid.

constrained_space = SearchSpace(
    batch_size=[32, 64, 128, 256],
    hidden_size=[128, 256, 512, 1024],
    num_layers=[2, 4, 6, 8],
)

# Memory limit: batch_size * hidden_size * num_layers < threshold
constrained_space.add_constraint(
    lambda p: p["batch_size"] * p["hidden_size"] * p["num_layers"] < 500_000,
    name="memory_limit",
)

# Large batches require smaller models
constrained_space.add_constraint(
    lambda p: not (p["batch_size"] >= 256 and p["hidden_size"] >= 512),
    name="large_batch_constraint",
)

print(f"Constrained space: {len(constrained_space.constraints)} constraints")


# Combining conditions and constraints
combined_space = (
    SearchSpace(
        model=["linear", "tree", "neural"],
        learning_rate=(1e-5, 1e-1, "log"),
        n_estimators=[10, 50, 100],
        hidden_layers=[1, 2, 3, 4],
    )
    .add_condition("n_estimators", when=lambda p: p["model"] == "tree")
    .add_condition("hidden_layers", when=lambda p: p["model"] == "neural")
    .add_condition("learning_rate", when=lambda p: p["model"] != "linear")
    .add_constraint(
        lambda p: not (
            p["model"] == "neural"
            and p.get("hidden_layers", 1) > 3
            and p.get("learning_rate", 0.01) > 0.01
        ),
        name="stable_training",
    )
)


# =============================================================================
# Nested Search Spaces
# =============================================================================

# Nested spaces allow hierarchical parameter structures.
# A dict with class/callable keys and dict values is automatically detected
# as a nested space.

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

estimator_space = SearchSpace(
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

print(
    f"Nested space: {len(estimator_space)} dimensions, "
    f"{len(estimator_space.conditions)} auto-conditions"
)


# Nested spaces with named functions
def log_transform(x):
    return np.log1p(x)


def sqrt_transform(x):
    return np.sqrt(x)


def identity(x):
    return x


transform_space = SearchSpace(
    preprocessor={
        log_transform: {"offset": (0.0, 1.0)},
        sqrt_transform: {"epsilon": (1e-8, 1e-4, "log")},
        identity: {},
    },
)


# =============================================================================
# Union of Search Spaces
# =============================================================================

# Combine search spaces using the | operator or union() method.

base_space = SearchSpace(
    learning_rate=(1e-5, 1e-1, "log"),
    batch_size=[32, 64, 128],
)

regularization_space = SearchSpace(
    weight_decay=(1e-6, 1e-2, "log"),
    dropout=(0.0, 0.5),
)

# Union with | operator
training_space = base_space | regularization_space

print(f"Union space: {list(training_space.param_names)}")


# Union with conflict resolution
space_a = SearchSpace(x=(0.0, 1.0), y=[1, 2, 3])
space_b = SearchSpace(x=(0.0, 10.0), z=["a", "b"])

# Default: "last" - use values from second space on conflict
merged_last = space_a.union(space_b, on_conflict="last")

# Keep first space values on conflict
merged_first = space_a.union(space_b, on_conflict="first")


# =============================================================================
# Dict-based Specification (backward compatibility)
# =============================================================================

# SearchSpace also accepts a dict for backward compatibility
dict_space = SearchSpace(
    {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
        "method": ["gradient", "newton", "bfgs"],
    }
)


# =============================================================================
# Inspection and Validation
# =============================================================================

# Check space properties
print(f"\nSpace inspection:")
print(f"  has_conditions: {svm_space.has_conditions}")
print(f"  has_constraints: {constrained_space.has_constraints}")
print(f"  param_names: {svm_space.param_names}")

# Get dimension types
types = svm_space.get_dimension_types()
for name, dim_type in types.items():
    print(f"  {name}: {dim_type.value}")

# Test constraint checking
test_params = {"batch_size": 64, "hidden_size": 256, "num_layers": 4}
is_valid = constrained_space.check_constraints(test_params)
print(f"\nConstraint check: {test_params} -> valid={is_valid}")

# Filter active parameters based on conditions
full_params = {"kernel": "rbf", "C": 1.0, "gamma": 0.1, "degree": 3}
active_params = svm_space.filter_active_params(full_params)
print(f"Active params for kernel='rbf': {list(active_params.keys())}")
