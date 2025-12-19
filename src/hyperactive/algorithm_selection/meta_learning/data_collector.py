"""Data collection for meta-learning.

This module runs optimization algorithms on benchmark problems
and collects performance data for training the algorithm selector.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np

from hyperactive.base import BaseOptimizer


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run.

    Parameters
    ----------
    name : str
        Human-readable name for this configuration.
    experiment_class : type
        The experiment class (e.g., Ackley, Sphere).
    experiment_params : dict
        Parameters to pass to experiment constructor.
    search_space : dict
        Search space for the optimization.
    n_iter : int
        Number of iterations.
    objective_func : Callable, optional
        The objective function (for AST feature extraction).
        If None, AST features cannot be extracted for this config.
    """

    name: str
    experiment_class: type
    experiment_params: dict
    search_space: dict
    n_iter: int
    objective_func: Optional[Callable] = None


@dataclass
class BenchmarkResult:
    """Result from running an optimizer on a benchmark.

    Parameters
    ----------
    config_name : str
        Name of the benchmark configuration.
    optimizer_name : str
        Name of the optimizer class.
    best_score : float
        Best score achieved.
    n_iter : int
        Number of iterations run.
    random_state : int
        Random seed used.
    """

    config_name: str
    optimizer_name: str
    best_score: float
    n_iter: int
    random_state: int
    extra: dict = field(default_factory=dict)


class BenchmarkDataCollector:
    """Collect performance data by running optimizers on benchmarks.

    This class runs multiple optimizers on multiple benchmark problems
    and records the results for training an algorithm selection model.

    Parameters
    ----------
    n_runs : int, default=3
        Number of runs per optimizer-benchmark combination.
    random_seed_base : int, default=42
        Base random seed (incremented for each run).
    verbose : bool, default=False
        Whether to print progress.

    Examples
    --------
    >>> collector = BenchmarkDataCollector(n_runs=2, verbose=False)
    >>> configs = collector.get_default_configs()
    >>> results = collector.collect(configs[:2])  # Run on subset
    """

    def __init__(
        self,
        n_runs: int = 3,
        random_seed_base: int = 42,
        verbose: bool = False,
    ):
        self.n_runs = n_runs
        self.random_seed_base = random_seed_base
        self.verbose = verbose

    def get_default_configs(self) -> list[BenchmarkConfig]:
        """Get default benchmark configurations.

        Returns a diverse set of benchmark configurations covering:
        - Different problem types (unimodal, multimodal)
        - Different dimensionalities
        - Different search space sizes

        Returns
        -------
        list of BenchmarkConfig
            Default benchmark configurations.
        """
        from hyperactive.experiment.bench import Ackley, Parabola, Sphere

        configs = []

        # Vary dimensions and resolution
        dimensions = [2, 5, 10]
        resolutions = [10, 25, 50]
        n_iters = [50, 100, 200]

        # Sphere (unimodal, convex)
        for n_dim in dimensions:
            for res in resolutions:
                for n_iter in n_iters:
                    search_space = {
                        f"x{i}": list(np.linspace(-5, 5, res))
                        for i in range(n_dim)
                    }
                    configs.append(
                        BenchmarkConfig(
                            name=f"sphere_d{n_dim}_r{res}_i{n_iter}",
                            experiment_class=Sphere,
                            experiment_params={"n_dim": n_dim},
                            search_space=search_space,
                            n_iter=n_iter,
                        )
                    )

        # Ackley (multimodal, non-convex)
        for n_dim in dimensions:
            for res in resolutions:
                for n_iter in n_iters:
                    search_space = {
                        f"x{i}": list(np.linspace(-5, 5, res))
                        for i in range(n_dim)
                    }
                    configs.append(
                        BenchmarkConfig(
                            name=f"ackley_d{n_dim}_r{res}_i{n_iter}",
                            experiment_class=Ackley,
                            experiment_params={"d": n_dim},
                            search_space=search_space,
                            n_iter=n_iter,
                        )
                    )

        # Parabola (simple 2D)
        for res in resolutions:
            for n_iter in n_iters:
                search_space = {
                    "x": list(np.linspace(-5, 5, res)),
                    "y": list(np.linspace(-5, 5, res)),
                }
                configs.append(
                    BenchmarkConfig(
                        name=f"parabola_r{res}_i{n_iter}",
                        experiment_class=Parabola,
                        experiment_params={},
                        search_space=search_space,
                        n_iter=n_iter,
                    )
                )

        return configs

    def get_synthetic_configs(
        self,
        n_iters: Optional[list[int]] = None,
        resolutions: Optional[list[int]] = None,
    ) -> list[BenchmarkConfig]:
        """Get synthetic benchmark configurations with AST-analyzable functions.

        These benchmarks use pure Python functions defined in the
        synthetic_benchmarks module, enabling AST feature extraction.

        Parameters
        ----------
        n_iters : list of int, optional
            Iteration budgets. Defaults to [50, 100, 200].
        resolutions : list of int, optional
            Search space resolutions. Defaults to [20].

        Returns
        -------
        list of BenchmarkConfig
            Synthetic benchmark configurations.
        """
        from hyperactive.experiment.func import FunctionExperiment

        from .synthetic_benchmarks import (
            get_all_benchmarks,
            get_search_space_for_benchmark,
        )

        if n_iters is None:
            n_iters = [50, 100, 200]
        if resolutions is None:
            resolutions = [20]

        configs = []
        benchmarks = get_all_benchmarks()

        for benchmark in benchmarks:
            for res in resolutions:
                search_space = get_search_space_for_benchmark(benchmark, res)
                for n_iter in n_iters:
                    config_name = f"{benchmark.name}_r{res}_i{n_iter}"
                    configs.append(
                        BenchmarkConfig(
                            name=config_name,
                            experiment_class=FunctionExperiment,
                            experiment_params={
                                "func": benchmark.func,
                            },
                            search_space=search_space,
                            n_iter=n_iter,
                            objective_func=benchmark.func,
                        )
                    )

        return configs

    def get_default_optimizers(self) -> list[Type[BaseOptimizer]]:
        """Get default list of optimizers to benchmark.

        Returns
        -------
        list of type
            Optimizer classes to benchmark.
        """
        from hyperactive.opt import (
            BayesianOptimizer,
            DifferentialEvolution,
            EvolutionStrategy,
            GeneticAlgorithm,
            HillClimbing,
            ParallelTempering,
            ParticleSwarmOptimizer,
            RandomRestartHillClimbing,
            RandomSearch,
            SimulatedAnnealing,
            StochasticHillClimbing,
        )

        return [
            RandomSearch,
            HillClimbing,
            StochasticHillClimbing,
            RandomRestartHillClimbing,
            SimulatedAnnealing,
            ParallelTempering,
            ParticleSwarmOptimizer,
            GeneticAlgorithm,
            EvolutionStrategy,
            DifferentialEvolution,
            BayesianOptimizer,
        ]

    def collect(
        self,
        configs: Optional[list[BenchmarkConfig]] = None,
        optimizers: Optional[list[Type[BaseOptimizer]]] = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks and collect results.

        Parameters
        ----------
        configs : list of BenchmarkConfig, optional
            Benchmark configurations to run. Defaults to get_default_configs().
        optimizers : list of type, optional
            Optimizer classes to benchmark. Defaults to get_default_optimizers().

        Returns
        -------
        list of BenchmarkResult
            Results from all benchmark runs.
        """
        if configs is None:
            configs = self.get_default_configs()
        if optimizers is None:
            optimizers = self.get_default_optimizers()

        results = []
        total = len(configs) * len(optimizers) * self.n_runs

        if self.verbose:
            print(f"Running {total} benchmark combinations...")

        count = 0
        for config in configs:
            for opt_class in optimizers:
                for run_idx in range(self.n_runs):
                    random_state = self.random_seed_base + run_idx

                    result = self._run_single(
                        config, opt_class, random_state
                    )
                    if result is not None:
                        results.append(result)

                    count += 1
                    if self.verbose and count % 50 == 0:
                        print(f"  Progress: {count}/{total}")

        if self.verbose:
            print(f"Completed {len(results)} successful runs")

        return results

    def _run_single(
        self,
        config: BenchmarkConfig,
        opt_class: Type[BaseOptimizer],
        random_state: int,
    ) -> Optional[BenchmarkResult]:
        """Run a single optimizer on a single benchmark.

        Parameters
        ----------
        config : BenchmarkConfig
            Benchmark configuration.
        opt_class : type
            Optimizer class.
        random_state : int
            Random seed.

        Returns
        -------
        BenchmarkResult or None
            Result if successful, None if failed.
        """
        # Suppress all warnings during optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # Create experiment
                experiment = config.experiment_class(**config.experiment_params)

                # Create optimizer
                optimizer = opt_class(
                    search_space=config.search_space,
                    n_iter=config.n_iter,
                    random_state=random_state,
                    experiment=experiment,
                )

                # Run optimization
                best_params = optimizer.solve()

                # Get best score
                best_score = getattr(optimizer, "best_score_", None)
                if best_score is None:
                    # Try to evaluate at best params
                    score_result = experiment.score(best_params)
                    # score() returns (score, extra_dict) tuple
                    if isinstance(score_result, tuple):
                        best_score = score_result[0]
                    else:
                        best_score = score_result

                return BenchmarkResult(
                    config_name=config.name,
                    optimizer_name=opt_class.__name__,
                    best_score=float(best_score),
                    n_iter=config.n_iter,
                    random_state=random_state,
                )

            except Exception as e:
                if self.verbose:
                    print(f"  Failed: {config.name} + {opt_class.__name__}: {e}")
                return None

    def collect_to_dataframe(
        self,
        configs: Optional[list[BenchmarkConfig]] = None,
        optimizers: Optional[list[Type[BaseOptimizer]]] = None,
    ):
        """Collect results and return as pandas DataFrame.

        Parameters
        ----------
        configs : list of BenchmarkConfig, optional
            Benchmark configurations.
        optimizers : list of type, optional
            Optimizer classes.

        Returns
        -------
        pd.DataFrame
            Results as a DataFrame.
        """
        import pandas as pd

        results = self.collect(configs, optimizers)

        return pd.DataFrame([
            {
                "config_name": r.config_name,
                "optimizer_name": r.optimizer_name,
                "best_score": r.best_score,
                "n_iter": r.n_iter,
                "random_state": r.random_state,
            }
            for r in results
        ])
