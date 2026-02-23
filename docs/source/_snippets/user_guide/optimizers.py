"""Optimizers page code snippets for documentation.

This snippet file contains examples from the optimizers.rst page covering
all optimizer categories and configurations.
"""

import numpy as np

# Define common test fixtures
search_space = {
    "x": np.arange(-5, 5, 0.5),
    "y": np.arange(-5, 5, 0.5),
}


def objective(params):
    x = params["x"]
    y = params["y"]
    return -(x**2 + y**2)


# ============================================================================
# Local Search Optimizers
# ============================================================================

# [start:hill_climbing]
from hyperactive.opt.gfo import HillClimbing

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:hill_climbing]


# [start:simulated_annealing]
from hyperactive.opt.gfo import SimulatedAnnealing

optimizer = SimulatedAnnealing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:simulated_annealing]


# [start:repulsing_hill_climbing]
from hyperactive.opt.gfo import RepulsingHillClimbing

optimizer = RepulsingHillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:repulsing_hill_climbing]


# [start:stochastic_hill_climbing]
from hyperactive.opt.gfo import StochasticHillClimbing

optimizer = StochasticHillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
    p_accept=0.3,  # Probability of accepting worse solutions
)
# [end:stochastic_hill_climbing]


# [start:downhill_simplex]
from hyperactive.opt.gfo import DownhillSimplexOptimizer

optimizer = DownhillSimplexOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:downhill_simplex]


# ============================================================================
# Global Search Optimizers
# ============================================================================

# [start:random_search]
from hyperactive.opt.gfo import RandomSearch

optimizer = RandomSearch(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:random_search]


# [start:grid_search]
from hyperactive.opt.gfo import GridSearch

optimizer = GridSearch(
    search_space=search_space,
    experiment=objective,
)
# [end:grid_search]


# [start:random_restart_hill_climbing]
from hyperactive.opt.gfo import RandomRestartHillClimbing

optimizer = RandomRestartHillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:random_restart_hill_climbing]


# [start:powells_pattern]
# [end:powells_pattern]


# ============================================================================
# Population Methods
# ============================================================================

# [start:particle_swarm]
from hyperactive.opt.gfo import ParticleSwarmOptimizer

optimizer = ParticleSwarmOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:particle_swarm]


# [start:genetic_algorithm]
from hyperactive.opt.gfo import GeneticAlgorithm

optimizer = GeneticAlgorithm(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:genetic_algorithm]


# [start:evolution_strategy]
# [end:evolution_strategy]


# [start:differential_evolution]
# [end:differential_evolution]


# [start:parallel_tempering]
# [end:parallel_tempering]


# [start:spiral_optimization]
# [end:spiral_optimization]


# ============================================================================
# Sequential Model-Based (Bayesian)
# ============================================================================

# [start:bayesian_optimizer]
from hyperactive.opt.gfo import BayesianOptimizer

optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:bayesian_optimizer]


# [start:tpe]
# [end:tpe]


# [start:forest_optimizer]
# [end:forest_optimizer]


# [start:lipschitz_direct]
# [end:lipschitz_direct]


# ============================================================================
# Optuna Backend
# ============================================================================

# [start:optuna_imports]
from hyperactive.opt.optuna import (
    TPEOptimizer,  # Tree-Parzen Estimators
)

# [end:optuna_imports]


# [start:optuna_tpe]

optimizer = TPEOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
)
# [end:optuna_tpe]


# ============================================================================
# Configuration Examples
# ============================================================================

# [start:common_parameters]
optimizer = SomeOptimizer(  # noqa: F821
    search_space=search_space,  # Required: parameter ranges
    n_iter=5,  # Required: number of iterations
    experiment=objective,  # Required: objective function
    random_state=42,  # Optional: for reproducibility
    initialize={  # Optional: initialization settings
        "warm_start": [...],  # Starting points
        "grid": 4,  # Grid initialization points
        "random": 2,  # Random initialization points
        "vertices": 4,  # Vertex initialization points
    },
)
# [end:common_parameters]


# [start:warm_start_example]
# Start from known good points
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
    initialize={
        "warm_start": [
            {"param1": 10, "param2": 0.5},
            {"param1": 20, "param2": 0.3},
        ]
    },
)
# [end:warm_start_example]


# [start:initialization_strategies]
# Mix of initialization strategies
optimizer = ParticleSwarmOptimizer(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
    initialize={
        "grid": 4,  # 4 points on a grid
        "random": 6,  # 6 random points
        "vertices": 4,  # 4 corner points
    },
)
# [end:initialization_strategies]


# [start:simulated_annealing_config]
from hyperactive.opt.gfo import SimulatedAnnealing

optimizer = SimulatedAnnealing(
    search_space=search_space,
    n_iter=5,
    experiment=objective,
    # Algorithm-specific parameters
    # (check API reference for available options)
)
# [end:simulated_annealing_config]


# --- Runnable test code below ---
if __name__ == "__main__":
    from hyperactive.opt.gfo import (
        BayesianOptimizer,
        GeneticAlgorithm,
        HillClimbing,
        ParticleSwarmOptimizer,
        RandomSearch,
        SimulatedAnnealing,
    )

    search_space = {
        "x": np.arange(-5, 5, 0.5),
        "y": np.arange(-5, 5, 0.5),
    }

    def objective(params):
        x = params["x"]
        y = params["y"]
        return -(x**2 + y**2)

    # Test a few optimizers
    optimizers_to_test = [
        ("HillClimbing", HillClimbing),
        ("SimulatedAnnealing", SimulatedAnnealing),
        ("RandomSearch", RandomSearch),
        ("BayesianOptimizer", BayesianOptimizer),
    ]

    for name, OptimizerClass in optimizers_to_test:
        if name == "BayesianOptimizer":
            optimizer = OptimizerClass(
                search_space=search_space,
                n_iter=5,
                experiment=objective,
            )
        else:
            optimizer = OptimizerClass(
                search_space=search_space,
                n_iter=5,
                experiment=objective,
            )
        best_params = optimizer.solve()
        assert "x" in best_params
        assert "y" in best_params
        print(f"{name} passed!")

    print("All optimizer snippets passed!")
