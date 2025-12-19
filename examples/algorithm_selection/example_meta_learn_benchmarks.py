"""Example: Verifying algorithm recommendations with Meta-Learn benchmarks.

This example shows how to use Meta-Learn's BenchmarkDataCollector to
run actual benchmarks and compare them against AlgorithmSelector's
recommendations.

Requirements:
    pip install meta-learn
"""

from hyperactive.algorithm_selection import AlgorithmSelector


def objective(params):
    """Simple quadratic function for benchmarking."""
    return params["x"] ** 2 + params["y"] ** 2


search_space = {
    "x": [i * 0.1 for i in range(-50, 51)],
    "y": [i * 0.1 for i in range(-50, 51)],
}


def main():
    # Get AlgorithmSelector's recommendation
    selector = AlgorithmSelector()
    recommended = selector.recommend(objective, search_space, n_iter=50)
    print(f"AlgorithmSelector recommends: {recommended.__name__}")

    # Verify with actual benchmarks using Meta-Learn
    try:
        from meta_learn.auto_opt.data_collection import (
            BenchmarkDataCollector,
            BenchmarkConfig,
        )
        from meta_learn.auto_opt.data_collection.adapters import get_hyperactive_optimizers
        import pandas as pd
    except ImportError:
        print("\nMeta-Learn not available. Install with: pip install meta-learn")
        return

    # Configure benchmark
    config = BenchmarkConfig(
        name="quadratic",
        objective=objective,
        search_space=search_space,
        n_iter=50,
    )

    # Select optimizers to benchmark
    all_optimizers = get_hyperactive_optimizers()
    test_optimizers = {
        k: v for k, v in all_optimizers.items()
        if k in ["RandomSearch", "HillClimbing", "SimulatedAnnealing",
                 "ParticleSwarmOptimizer", "BayesianOptimizer"]
    }

    # Run benchmarks
    print("\nRunning benchmarks (5 runs per optimizer)...")
    collector = BenchmarkDataCollector(n_runs=5, verbose=False)
    results = collector.collect([config], test_optimizers)

    # Analyze results
    df = pd.DataFrame([
        {"optimizer": r.optimizer_name, "score": r.best_score}
        for r in results
    ])
    mean_scores = df.groupby("optimizer")["score"].mean().sort_values()

    print("\nActual performance (lower = better):")
    for opt, score in mean_scores.items():
        print(f"  {opt}: {score:.4f}")

    actual_best = mean_scores.idxmin()
    match = "MATCH" if recommended.__name__ == actual_best else "DIFFERENT"
    print(f"\nRecommended: {recommended.__name__}, Actual best: {actual_best} ({match})")


if __name__ == "__main__":
    main()
