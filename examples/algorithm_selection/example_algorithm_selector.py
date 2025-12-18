"""Example: Using AlgorithmSelector to get optimizer recommendations.

AlgorithmSelector analyzes your problem and returns ranked recommendations.
This gives you more control than AutoOptimizer - you can inspect the
recommendations and choose your own optimizer.
"""

import math

from hyperactive.algorithm_selection import AlgorithmSelector


def main():
    # Define different types of objective functions
    # to see how recommendations change

    # 1. Simple quadratic (unimodal, smooth)
    def sphere(params):
        return params["x"] ** 2 + params["y"] ** 2

    # 2. Multimodal function with trigonometric components
    def rastrigin_like(params):
        x, y = params["x"], params["y"]
        return x**2 + y**2 + 10 * (2 - math.cos(2 * math.pi * x) - math.cos(2 * math.pi * y))

    # 3. Function with conditionals (potentially discontinuous)
    def conditional_objective(params):
        x, y = params["x"], params["y"]
        if x > 0 and y > 0:
            return x**2 + y**2
        elif x < 0 and y < 0:
            return (x + 1) ** 2 + (y + 1) ** 2
        else:
            return x**2 + y**2 + 10

    # Search space
    search_space = {
        "x": [i * 0.1 for i in range(-50, 51)],
        "y": [i * 0.1 for i in range(-50, 51)],
    }

    selector = AlgorithmSelector()

    print("=" * 60)
    print("Algorithm Selection Recommendations")
    print("=" * 60)

    # Analyze each function
    for name, func in [
        ("Sphere (simple quadratic)", sphere),
        ("Rastrigin-like (multimodal)", rastrigin_like),
        ("Conditional (discontinuous)", conditional_objective),
    ]:
        print(f"\n{name}")
        print("-" * 40)

        rankings = selector.rank(func, search_space, n_iter=100)

        # Show top 5 recommendations
        print("Top 5 recommended optimizers:")
        for i, (opt_class, score) in enumerate(list(rankings.items())[:5], 1):
            print(f"  {i}. {opt_class.__name__}: {score:.3f}")

        # Show extracted features
        print(f"\nExtracted features:")
        print(f"  - Trigonometric functions: {selector.ast_features_.num_sin + selector.ast_features_.num_cos}")
        print(f"  - Power operations: {selector.ast_features_.num_pow}")
        print(f"  - Conditionals: {selector.ast_features_.num_if}")
        print(f"  - Search space size: {selector.search_space_features_.total_size}")

    # Example: Using the recommendation
    print("\n" + "=" * 60)
    print("Using the recommendation")
    print("=" * 60)

    recommendation = selector.recommend(sphere, search_space, n_iter=100)
    print(f"\nRecommended optimizer for Sphere: {recommendation.__name__}")

    # Instantiate and run
    optimizer = recommendation(
        experiment=lambda x: -sphere(x),  # Negate for minimization
        search_space=search_space,
        n_iter=50,
    )
    best_params = optimizer.solve()
    print(f"Best parameters found: x={best_params['x']:.3f}, y={best_params['y']:.3f}")
    print(f"Best score: {sphere(best_params):.6f}")


if __name__ == "__main__":
    main()
