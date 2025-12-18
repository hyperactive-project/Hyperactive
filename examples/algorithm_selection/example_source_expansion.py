"""Example: Source expansion for objective functions with helper functions.

When your objective function calls other functions, the algorithm selector
can "expand" the source code to analyze the full computation. This helps
it make better recommendations.
"""

import math

from hyperactive.algorithm_selection import AlgorithmSelector
from hyperactive.algorithm_selection.ast_feature_engineering import (
    SourceExpander,
    extract_ast_features,
)


# Define helper functions
def compute_distance(x, y):
    """Compute Euclidean distance from origin."""
    return math.sqrt(x**2 + y**2)


def add_penalty(value, threshold=5.0):
    """Add a penalty for values exceeding threshold."""
    if value > threshold:
        return value + (value - threshold) ** 2
    return value


def apply_modulation(value, freq=1.0):
    """Apply sinusoidal modulation."""
    return value * (1 + 0.5 * math.sin(freq * value))


# Main objective function that uses helpers
def objective(params):
    """Complex objective that calls multiple helper functions."""
    x, y = params["x"], params["y"]

    # Use helper functions
    dist = compute_distance(x, y)
    penalized = add_penalty(dist)
    modulated = apply_modulation(penalized)

    return -modulated  # Negate for minimization


def main():
    search_space = {
        "x": [i * 0.2 for i in range(-25, 26)],
        "y": [i * 0.2 for i in range(-25, 26)],
    }

    print("=" * 60)
    print("Source Expansion Example")
    print("=" * 60)

    # Show the source expansion
    print("\n1. Source Expansion")
    print("-" * 40)

    expander = SourceExpander()
    expanded_source = expander.expand(objective)

    print("Expanded source includes:")
    for entry in expander.resolution_log:
        if entry["success"]:
            print(f"  - {entry['function']}")

    print(f"\nTotal expanded source length: {len(expanded_source)} characters")

    # Compare features with and without expansion
    print("\n2. Feature Comparison")
    print("-" * 40)

    features_expanded = extract_ast_features(objective, expand_source=True)
    features_not_expanded = extract_ast_features(objective, expand_source=False)

    print("Features WITHOUT source expansion:")
    print(f"  - num_sin: {features_not_expanded.num_sin}")
    print(f"  - num_sqrt: {features_not_expanded.num_sqrt}")
    print(f"  - num_if: {features_not_expanded.num_if}")
    print(f"  - num_pow: {features_not_expanded.num_pow}")

    print("\nFeatures WITH source expansion:")
    print(f"  - num_sin: {features_expanded.num_sin}")
    print(f"  - num_sqrt: {features_expanded.num_sqrt}")
    print(f"  - num_if: {features_expanded.num_if}")
    print(f"  - num_pow: {features_expanded.num_pow}")

    # Show how this affects recommendations
    print("\n3. Impact on Recommendations")
    print("-" * 40)

    selector_expanded = AlgorithmSelector(expand_source=True)
    selector_not_expanded = AlgorithmSelector(expand_source=False)

    rankings_expanded = selector_expanded.rank(objective, search_space)
    rankings_not_expanded = selector_not_expanded.rank(objective, search_space)

    print("Top 3 recommendations WITHOUT expansion:")
    for i, (opt, score) in enumerate(list(rankings_not_expanded.items())[:3], 1):
        print(f"  {i}. {opt.__name__}: {score:.3f}")

    print("\nTop 3 recommendations WITH expansion:")
    for i, (opt, score) in enumerate(list(rankings_expanded.items())[:3], 1):
        print(f"  {i}. {opt.__name__}: {score:.3f}")

    print("\n" + "=" * 60)
    print("Note: With source expansion, the selector can detect")
    print("trigonometric functions (sin) and conditionals (if)")
    print("in helper functions, leading to potentially different")
    print("and more informed recommendations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
