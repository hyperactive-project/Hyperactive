"""Example: Using AutoOptimizer for automatic algorithm selection.

AutoOptimizer analyzes your objective function and search space to
automatically select the best optimization algorithm for your problem.
"""

from hyperactive.algorithm_selection import AutoOptimizer


def main():
    # Define a simple objective function
    def objective(params):
        """Sphere function - a simple convex optimization problem."""
        x = params["x"]
        y = params["y"]
        return -(x**2 + y**2)  # Negative because we maximize

    # Define the search space
    search_space = {
        "x": [i * 0.1 for i in range(-50, 51)],  # -5.0 to 5.0
        "y": [i * 0.1 for i in range(-50, 51)],
    }

    # Create and run AutoOptimizer
    auto = AutoOptimizer(
        experiment=objective,
        search_space=search_space,
        n_iter=100,
        verbose=True,  # Print which algorithm was selected
    )

    # Run the optimization
    best_params = auto.solve()

    # Results
    print("\n--- Results ---")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {objective(best_params)}")
    print(f"Selected optimizer: {auto.selected_optimizer_.__name__}")

    # You can also inspect the full ranking
    print("\n--- Full Rankings ---")
    for opt_class, score in list(auto.selection_scores_.items())[:5]:
        print(f"  {opt_class.__name__}: {score:.3f}")


if __name__ == "__main__":
    main()
