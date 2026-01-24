# GFO (Gradient-Free Optimization) Examples

This directory contains examples demonstrating the GFO optimization backend in Hyperactive. GFO provides a collection of gradient-free optimization algorithms that don't require derivative information, making them suitable for black-box optimization problems like hyperparameter tuning.

## Quick Start

Run any example directly:
```bash
python random_search_example.py
python hill_climbing_example.py
python bayesian_optimization_example.py
# ... etc
```

## Available Algorithms

| Algorithm | Type | Best For | Characteristics |
|-----------|------|----------|-----------------|
| [RandomSearch](random_search_example.py) | Random | Baselines, high-dim spaces | Simple, parallel-friendly |
| [GridSearch](grid_search_example.py) | Exhaustive | Small discrete spaces | Complete coverage |
| [HillClimbing](hill_climbing_example.py) | Local Search | Fast convergence | Exploits local structure |
| [StochasticHillClimbing](stochastic_hill_climbing_example.py) | Local Search | Plateau handling | Probabilistic moves |
| [RepulsingHillClimbing](repulsing_hill_climbing_example.py) | Local Search | Multi-optima discovery | Memory-based repulsion |
| [RandomRestartHillClimbing](random_restart_hill_climbing_example.py) | Multi-start | Multi-modal problems | Multiple hill climbs |
| [SimulatedAnnealing](simulated_annealing_example.py) | Probabilistic | Escaping local optima | Temperature cooling |
| [ParallelTempering](parallel_tempering_example.py) | MCMC | Complex landscapes | Multi-temperature chains |
| [PatternSearch](pattern_search_example.py) | Direct Search | Noisy functions | Systematic patterns |
| [PowellsMethod](powells_method_example.py) | Coordinate Descent | Smooth functions | Conjugate directions |
| [DownhillSimplexOptimizer](downhill_simplex_example.py) | Simplex | Low-dimensional | Nelder-Mead method |
| [DirectAlgorithm](direct_algorithm_example.py) | Global | Lipschitz functions | Space division |
| [LipschitzOptimizer](lipschitz_optimizer_example.py) | Global | Smooth functions | Theoretical guarantees |
| [SpiralOptimization](spiral_optimization_example.py) | Nature-inspired | Mixed problems | Spiral search patterns |
| [ParticleSwarmOptimizer](particle_swarm_example.py) | Swarm Intelligence | Continuous problems | Population-based |
| [GeneticAlgorithm](genetic_algorithm_example.py) | Evolutionary | Multi-modal problems | Selection/crossover/mutation |
| [EvolutionStrategy](evolution_strategy_example.py) | Evolutionary | Self-adaptive | Strategy parameter evolution |
| [DifferentialEvolution](differential_evolution_example.py) | Evolutionary | Continuous problems | Vector differences |
| [BayesianOptimizer](bayesian_optimization_example.py) | Probabilistic | Expensive evaluations | Gaussian Process |
| [TreeStructuredParzenEstimators](tree_structured_parzen_estimators_example.py) | Bayesian | Mixed parameters | TPE algorithm |
| [ForestOptimizer](forest_optimizer_example.py) | Surrogate-based | Non-linear problems | Random Forest surrogates |
| [Union Grid Search Space Example](search_space_union_example.py) | Utility | Grid unions | Union grids + array-like values |

## Algorithm Categories

### Simple Algorithms
- **RandomSearch**: Pure random sampling - excellent baseline
- **GridSearch**: Exhaustive enumeration of discrete grids

### Local Search Methods
- **HillClimbing**: Greedy local search from starting points
- **SimulatedAnnealing**: Hill climbing with probabilistic escapes
- **StochasticHillClimbing**: Hill climbing with random moves

### Population-Based Methods
- **ParticleSwarmOptimizer**: Swarm intelligence optimization
- **GeneticAlgorithm**: Evolutionary computation approach
- **DifferentialEvolution**: Evolution strategy for continuous spaces

### Advanced Methods
- **BayesianOptimizer**: Gaussian Process-based optimization
- **TreeStructuredParzenEstimators**: TPE algorithm (similar to Optuna's TPE)
- **ForestOptimizer**: Random forest-based surrogate optimization

## Choosing the Right Algorithm

### Quick Decision Tree

1. **Need a baseline?** → RandomSearch
2. **Small discrete space?** → GridSearch
3. **Expensive evaluations?** → BayesianOptimizer
4. **Continuous optimization?** → ParticleSwarmOptimizer or BayesianOptimizer
5. **Fast local optimization?** → HillClimbing
6. **General purpose?** → RandomSearch or BayesianOptimizer

### Problem Characteristics

**Function Properties:**
- Smooth, continuous → BayesianOptimizer, ParticleSwarmOptimizer
- Noisy, discontinuous → RandomSearch, GeneticAlgorithm
- Multi-modal → ParticleSwarmOptimizer, GeneticAlgorithm

**Search Space:**
- Low dimensional (< 10) → GridSearch, BayesianOptimizer
- High dimensional (> 20) → RandomSearch, GeneticAlgorithm
- Mixed types → RandomSearch, TreeStructuredParzenEstimators

**Computational Budget:**
- Low budget (< 50 trials) → RandomSearch, HillClimbing
- Medium budget (50-200) → BayesianOptimizer, ParticleSwarmOptimizer
- High budget (200+) → GridSearch, GeneticAlgorithm

## Common Configuration

All GFO algorithms support:
- **Random state**: `random_state=42` for reproducibility
- **Early stopping**: `early_stopping=10` trials without improvement
- **Max score**: `max_score=0.99` stop when target reached
- **Warm start**: `initialize={"warm_start": [points]}` initial solutions

## Advanced Usage

### Sequential Optimization
```python
# Phase 1: Global exploration
random_opt = RandomSearch(n_trials=20, ...)
initial_results = random_opt.solve()

# Phase 2: Local refinement
hill_opt = HillClimbing(
    n_trials=30,
    initialize={"warm_start": [initial_results]}
)
final_results = hill_opt.solve()
```

### Parameter Space Design

**Continuous Spaces:**
```python
param_space = {
    "learning_rate": (0.001, 0.1),      # Log scale recommended
    "n_estimators": (10, 1000),         # Integer range
    "regularization": (0.0, 1.0),       # Bounded continuous
}
```

**Discrete/Categorical:**
```python
param_space = {
    "algorithm": ["adam", "sgd", "rmsprop"],  # Categorical
    "layers": [1, 2, 3, 4, 5],               # Discrete integers
    "activation": ["relu", "tanh", "sigmoid"], # Categorical
}
```

## Performance Tips

1. **Start simple**: RandomSearch is often surprisingly competitive
2. **Use warm starts**: Provide good initial points when available
3. **Set early stopping**: Avoid wasted evaluations on poor runs
4. **Match algorithm to problem**: Consider function smoothness and dimensionality
5. **Reproducibility**: Always set `random_state` for consistent results

## Further Reading

- [Gradient-Free Optimization Overview](https://en.wikipedia.org/wiki/Derivative-free_optimization)
- [Bayesian Optimization Tutorial](https://arxiv.org/abs/1807.02811)
- [Evolutionary Algorithms Survey](https://ieeexplore.ieee.org/document/6900297)
- [Hyperparameter Optimization Review](https://arxiv.org/abs/1502.02127)
