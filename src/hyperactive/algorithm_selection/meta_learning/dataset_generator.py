"""Dataset generation for meta-learning pipeline.

This module converts raw benchmark results into training data with
feature vectors and pairwise ranking labels.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import inspect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hyperactive.algorithm_selection.ast_feature_engineering import (
    ASTFeatureExtractor,
    SearchSpaceFeatureExtractor,
)
from hyperactive.base import BaseExperiment

from .data_collector import BenchmarkConfig, RunResult


# Fixed optimizer order for consistent one-hot encoding
OPTIMIZER_NAMES = [
    "RandomSearch",
    "HillClimbing",
    "StochasticHillClimbing",
    "RandomRestartHillClimbing",
    "SimulatedAnnealing",
    "ParallelTempering",
    "ParticleSwarmOptimizer",
    "GeneticAlgorithm",
    "EvolutionStrategy",
    "DifferentialEvolution",
    "BayesianOptimizer",
]


@dataclass
class TrainingSample:
    """A single pairwise training sample.

    Parameters
    ----------
    features : list of float
        Combined problem feature vector.
    optimizer_a_idx : int
        Index of first optimizer in OPTIMIZER_NAMES.
    optimizer_b_idx : int
        Index of second optimizer in OPTIMIZER_NAMES.
    label : int
        1 if optimizer A beats optimizer B, 0 otherwise.
    """

    features: list[float]
    optimizer_a_idx: int
    optimizer_b_idx: int
    label: int


class RankingDatasetGenerator:
    """Generate pairwise ranking training data from benchmark results.

    This class converts raw benchmark results into a dataset suitable
    for training a pairwise ranking model. For each problem instance,
    it generates all pairs of optimizers with labels indicating which
    optimizer performs better.

    Examples
    --------
    >>> generator = RankingDatasetGenerator()
    >>> results = BenchmarkDataCollector.load_results("results.json")
    >>> X, y = generator.generate_dataset(results)
    >>> X.shape  # (n_samples, n_features)
    (3465, 68)
    """

    def __init__(self):
        self.ast_extractor = ASTFeatureExtractor(expand_source=False)
        self.ss_extractor = SearchSpaceFeatureExtractor()

    def extract_problem_features(
        self,
        benchmark: BaseExperiment,
        search_space: dict[str, list],
        n_iter: int,
    ) -> list[float]:
        """Extract combined feature vector for a problem instance.

        Parameters
        ----------
        benchmark : BaseExperiment
            The benchmark instance.
        search_space : dict
            The search space dictionary.
        n_iter : int
            Number of iterations.

        Returns
        -------
        list of float
            Combined feature vector (AST + search space + n_iter).
        """
        # Extract AST features from the benchmark's evaluate method
        try:
            source = inspect.getsource(benchmark._evaluate)
            ast_features = self.ast_extractor.extract_from_source(source)
        except (TypeError, OSError):
            # Fallback to empty features if source not available
            from hyperactive.algorithm_selection.ast_feature_engineering import (
                ASTFeatures,
            )

            ast_features = ASTFeatures()

        # Extract search space features
        ss_features = self.ss_extractor.extract(search_space)

        # Combine all features
        features = ast_features.to_vector() + ss_features.to_vector() + [n_iter]

        return features

    def scores_to_ranking(
        self, scores: dict[str, list[float]]
    ) -> dict[str, int]:
        """Convert optimizer scores to rankings.

        For minimization problems (lower is better), ranks optimizers
        from best (rank 1) to worst (rank N).

        Parameters
        ----------
        scores : dict
            Mapping from optimizer name to list of scores across runs.

        Returns
        -------
        dict
            Mapping from optimizer name to rank (1 = best).
        """
        # Average scores across runs
        avg_scores = {name: np.mean(runs) for name, runs in scores.items()}

        # Sort by score ascending (lower is better for benchmarks)
        sorted_opts = sorted(avg_scores.items(), key=lambda x: x[1])

        # Assign ranks (handle ties by giving same rank)
        rankings = {}
        current_rank = 1
        prev_score = None

        for i, (name, score) in enumerate(sorted_opts):
            if prev_score is not None and score > prev_score + 1e-10:
                current_rank = i + 1
            rankings[name] = current_rank
            prev_score = score

        return rankings

    def generate_pairwise_samples(
        self,
        features: list[float],
        rankings: dict[str, int],
    ) -> list[TrainingSample]:
        """Generate all pairwise comparisons for one problem instance.

        Parameters
        ----------
        features : list of float
            Problem feature vector.
        rankings : dict
            Mapping from optimizer name to rank.

        Returns
        -------
        list of TrainingSample
            Pairwise comparison samples (C(n,2) pairs for n optimizers).
        """
        samples = []
        n_opts = len(OPTIMIZER_NAMES)

        for i in range(n_opts):
            for j in range(i + 1, n_opts):
                name_a = OPTIMIZER_NAMES[i]
                name_b = OPTIMIZER_NAMES[j]

                # Skip if either optimizer not in rankings
                if name_a not in rankings or name_b not in rankings:
                    continue

                rank_a = rankings[name_a]
                rank_b = rankings[name_b]

                # Label: 1 if A is better (lower rank), 0 otherwise
                # For ties (same rank), label is 0 (arbitrary but consistent)
                label = 1 if rank_a < rank_b else 0

                samples.append(
                    TrainingSample(
                        features=features,
                        optimizer_a_idx=i,
                        optimizer_b_idx=j,
                        label=label,
                    )
                )

        return samples

    def _create_benchmark_from_config(
        self, benchmark_name: str, dimensions: int
    ) -> BaseExperiment:
        """Create benchmark instance from name and dimensions.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark class.
        dimensions : int
            Number of dimensions.

        Returns
        -------
        BaseExperiment
            The benchmark instance.
        """
        from hyperactive.experiment.bench import Ackley, Parabola, Sphere

        if benchmark_name == "Ackley":
            return Ackley(d=dimensions)
        elif benchmark_name == "Sphere":
            return Sphere(n_dim=dimensions)
        elif benchmark_name == "Parabola":
            return Parabola()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def _create_search_space(
        self,
        benchmark: BaseExperiment,
        resolution: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
    ) -> dict[str, list]:
        """Create search space for a benchmark.

        Parameters
        ----------
        benchmark : BaseExperiment
            The benchmark instance.
        resolution : int
            Number of values per dimension.
        bounds : tuple
            (min, max) bounds for values.

        Returns
        -------
        dict
            Search space dictionary.
        """
        param_names = benchmark.paramnames()
        values = list(np.linspace(bounds[0], bounds[1], resolution))
        return {name: values for name in param_names}

    def generate_dataset(
        self, results: list[RunResult]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset from benchmark results.

        Parameters
        ----------
        results : list of RunResult
            Raw benchmark results.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            Features are: problem features + optimizer_a one-hot + optimizer_b one-hot.
        y : np.ndarray
            Labels of shape (n_samples,), binary (0 or 1).
        """
        # Group results by config_id
        results_by_config: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        config_metadata: dict[str, dict] = {}

        for result in results:
            results_by_config[result.config_id][result.optimizer_name].append(
                result.best_score
            )
            # Store metadata for feature extraction
            if result.config_id not in config_metadata:
                config_metadata[result.config_id] = {
                    "benchmark_name": result.benchmark_name,
                    "dimensions": result.dimensions,
                    "search_space_resolution": result.search_space_resolution,
                    "n_iter": result.n_iter,
                }

        all_samples: list[TrainingSample] = []

        for config_id, optimizer_scores in results_by_config.items():
            meta = config_metadata[config_id]

            # Create benchmark and search space for feature extraction
            benchmark = self._create_benchmark_from_config(
                meta["benchmark_name"], meta["dimensions"]
            )
            search_space = self._create_search_space(
                benchmark, meta["search_space_resolution"]
            )

            # Extract features
            features = self.extract_problem_features(
                benchmark, search_space, meta["n_iter"]
            )

            # Convert scores to rankings
            rankings = self.scores_to_ranking(optimizer_scores)

            # Generate pairwise samples
            samples = self.generate_pairwise_samples(features, rankings)
            all_samples.extend(samples)

        # Convert to numpy arrays
        n_samples = len(all_samples)
        n_problem_features = len(all_samples[0].features) if all_samples else 0
        n_optimizers = len(OPTIMIZER_NAMES)

        # Feature vector: problem features + opt_a one-hot + opt_b one-hot
        n_features = n_problem_features + 2 * n_optimizers

        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=np.int32)

        for i, sample in enumerate(all_samples):
            # Problem features
            X[i, :n_problem_features] = sample.features

            # Optimizer A one-hot
            X[i, n_problem_features + sample.optimizer_a_idx] = 1

            # Optimizer B one-hot
            X[i, n_problem_features + n_optimizers + sample.optimizer_b_idx] = 1

            # Label
            y[i] = sample.label

        return X, y

    def get_feature_names(self) -> list[str]:
        """Get names of all features in the dataset.

        Returns
        -------
        list of str
            Feature names in order.
        """
        from hyperactive.algorithm_selection.ast_feature_engineering import (
            ASTFeatures,
            SearchSpaceFeatures,
        )

        # AST feature names
        ast_names = ASTFeatures.feature_names()

        # Search space feature names
        ss_names = SearchSpaceFeatures.feature_names()

        # n_iter
        problem_names = ast_names + ss_names + ["n_iter"]

        # Optimizer one-hot names
        opt_a_names = [f"opt_a_{name}" for name in OPTIMIZER_NAMES]
        opt_b_names = [f"opt_b_{name}" for name in OPTIMIZER_NAMES]

        return problem_names + opt_a_names + opt_b_names

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        path: str,
        save_feature_names: bool = True,
    ):
        """Save dataset to numpy files.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.
        path : str
            Base path (without extension).
        save_feature_names : bool, default=True
            Whether to save feature names to a text file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.save(f"{path}_X.npy", X)
        np.save(f"{path}_y.npy", y)

        if save_feature_names:
            with open(f"{path}_features.txt", "w") as f:
                for name in self.get_feature_names():
                    f.write(f"{name}\n")

    @staticmethod
    def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load dataset from numpy files.

        Parameters
        ----------
        path : str
            Base path (without extension).

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.
        """
        X = np.load(f"{path}_X.npy")
        y = np.load(f"{path}_y.npy")
        return X, y
