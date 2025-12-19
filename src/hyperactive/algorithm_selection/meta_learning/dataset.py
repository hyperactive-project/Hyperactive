"""Dataset generation for algorithm selection training.

This module transforms benchmark results into feature vectors and labels
for training the algorithm selection model.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..ast_feature_engineering import (
    ASTFeatureExtractor,
    ASTFeatures,
    SearchSpaceFeatureExtractor,
)
from .data_collector import BenchmarkConfig, BenchmarkResult


@dataclass
class TrainingExample:
    """A single training example for algorithm selection.

    Parameters
    ----------
    config_name : str
        Name of the benchmark configuration.
    features : np.ndarray
        Feature vector.
    best_optimizer : str
        Name of the best optimizer for this problem.
    optimizer_scores : dict
        All optimizer scores for this configuration.
    """

    config_name: str
    features: np.ndarray
    best_optimizer: str
    optimizer_scores: dict[str, float]


class DatasetGenerator:
    """Generate training datasets from benchmark results.

    This class processes benchmark results to create:
    - Feature vectors from search space and AST characteristics
    - Labels (best optimizer) from performance comparison

    Parameters
    ----------
    aggregation : str, default="mean"
        How to aggregate multiple runs: "mean", "median", or "best".
    include_ast_features : bool, default=True
        Whether to include AST features when objective_func is available.

    Examples
    --------
    >>> from hyperactive.algorithm_selection.meta_learning import (
    ...     BenchmarkDataCollector, DatasetGenerator
    ... )
    >>> collector = BenchmarkDataCollector(n_runs=2)
    >>> results = collector.collect(configs[:5])  # Small subset
    >>> generator = DatasetGenerator()
    >>> X, y, feature_names = generator.generate(results, configs[:5])
    """

    def __init__(self, aggregation: str = "mean", include_ast_features: bool = True):
        if aggregation not in ("mean", "median", "best"):
            raise ValueError("aggregation must be 'mean', 'median', or 'best'")
        self.aggregation = aggregation
        self.include_ast_features = include_ast_features

        self._ss_extractor = SearchSpaceFeatureExtractor()
        self._ast_extractor = ASTFeatureExtractor(expand_source=False)

    def generate(
        self,
        results: list[BenchmarkResult],
        configs: list[BenchmarkConfig],
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Generate training data from benchmark results.

        Parameters
        ----------
        results : list of BenchmarkResult
            Results from running optimizers on benchmarks.
        configs : list of BenchmarkConfig
            Benchmark configurations.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label array of shape (n_samples,) with optimizer indices.
        feature_names : list of str
            Names of features.
        optimizer_names : list of str
            Names of optimizers (for decoding y).
        """
        # Group results by config
        results_by_config = defaultdict(list)
        for r in results:
            results_by_config[r.config_name].append(r)

        # Build config lookup
        config_lookup = {c.name: c for c in configs}

        # Get all unique optimizers
        all_optimizers = sorted(set(r.optimizer_name for r in results))
        optimizer_to_idx = {name: i for i, name in enumerate(all_optimizers)}

        # Generate examples
        examples = []
        for config_name, config_results in results_by_config.items():
            if config_name not in config_lookup:
                continue

            config = config_lookup[config_name]

            # Aggregate scores per optimizer
            scores_by_opt = self._aggregate_scores(config_results)

            if not scores_by_opt:
                continue

            # Find best optimizer (highest score)
            best_opt = max(scores_by_opt, key=scores_by_opt.get)

            # Extract features
            features = self._extract_features(config)

            examples.append(
                TrainingExample(
                    config_name=config_name,
                    features=features,
                    best_optimizer=best_opt,
                    optimizer_scores=scores_by_opt,
                )
            )

        if not examples:
            raise ValueError("No valid examples generated from results")

        # Build arrays
        feature_names = self._get_feature_names()
        X = np.array([ex.features for ex in examples])
        y = np.array([optimizer_to_idx[ex.best_optimizer] for ex in examples])

        return X, y, feature_names, all_optimizers

    def _aggregate_scores(
        self, results: list[BenchmarkResult]
    ) -> dict[str, float]:
        """Aggregate scores from multiple runs per optimizer.

        Parameters
        ----------
        results : list of BenchmarkResult
            Results for a single configuration.

        Returns
        -------
        dict
            Optimizer name to aggregated score.
        """
        scores_by_opt = defaultdict(list)
        for r in results:
            scores_by_opt[r.optimizer_name].append(r.best_score)

        aggregated = {}
        for opt_name, scores in scores_by_opt.items():
            if self.aggregation == "mean":
                aggregated[opt_name] = np.mean(scores)
            elif self.aggregation == "median":
                aggregated[opt_name] = np.median(scores)
            else:  # best
                aggregated[opt_name] = np.max(scores)

        return aggregated

    def _extract_features(self, config: BenchmarkConfig) -> np.ndarray:
        """Extract features from a benchmark configuration.

        Parameters
        ----------
        config : BenchmarkConfig
            Benchmark configuration.

        Returns
        -------
        np.ndarray
            Feature vector.
        """
        # Search space features
        ss_features = self._ss_extractor.extract(config.search_space)

        # Base features
        features = [
            # Search space features
            ss_features.n_dimensions,
            ss_features.total_size,
            ss_features.n_continuous,
            ss_features.n_discrete,
            ss_features.n_categorical,
            ss_features.avg_choices_per_dim,
            ss_features.min_choices,
            ss_features.max_choices,
            ss_features.avg_range_span,
            1.0 if ss_features.has_mixed_types else 0.0,
            # Iteration budget
            config.n_iter,
        ]

        # AST features (if objective function is available)
        if self.include_ast_features and config.objective_func is not None:
            ast_features = self._ast_extractor.extract(config.objective_func)
            features.extend(ast_features.to_vector())
        elif self.include_ast_features:
            # Pad with zeros if no objective function available
            features.extend([0.0] * len(ASTFeatures.feature_names()))

        return np.array(features, dtype=np.float32)

    def _get_feature_names(self) -> list[str]:
        """Get feature names.

        Returns
        -------
        list of str
            Feature names in order.
        """
        base_names = [
            "n_dimensions",
            "total_size",
            "n_continuous",
            "n_discrete",
            "n_categorical",
            "avg_choices_per_dim",
            "min_choices",
            "max_choices",
            "avg_range_span",
            "has_mixed_types",
            "n_iter",
        ]

        if self.include_ast_features:
            # Add AST feature names prefixed with 'ast_'
            ast_names = [f"ast_{name}" for name in ASTFeatures.feature_names()]
            base_names.extend(ast_names)

        return base_names

    def generate_with_rankings(
        self,
        results: list[BenchmarkResult],
        configs: list[BenchmarkConfig],
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Generate training data with full optimizer rankings.

        This variant returns normalized ranking scores for each optimizer,
        which can be used for learning-to-rank approaches.

        Parameters
        ----------
        results : list of BenchmarkResult
            Results from running optimizers on benchmarks.
        configs : list of BenchmarkConfig
            Benchmark configurations.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y_rankings : np.ndarray
            Ranking matrix of shape (n_samples, n_optimizers).
            Each row contains normalized scores (0-1) for each optimizer.
        feature_names : list of str
            Names of features.
        optimizer_names : list of str
            Names of optimizers (column order for y_rankings).
        """
        # Group results by config
        results_by_config = defaultdict(list)
        for r in results:
            results_by_config[r.config_name].append(r)

        # Build config lookup
        config_lookup = {c.name: c for c in configs}

        # Get all unique optimizers
        all_optimizers = sorted(set(r.optimizer_name for r in results))
        n_optimizers = len(all_optimizers)

        # Generate examples
        X_list = []
        y_list = []

        for config_name, config_results in results_by_config.items():
            if config_name not in config_lookup:
                continue

            config = config_lookup[config_name]

            # Aggregate scores per optimizer
            scores_by_opt = self._aggregate_scores(config_results)

            if not scores_by_opt:
                continue

            # Extract features
            features = self._extract_features(config)

            # Build ranking vector (normalized scores)
            scores = np.zeros(n_optimizers)
            for i, opt_name in enumerate(all_optimizers):
                scores[i] = scores_by_opt.get(opt_name, float("-inf"))

            # Normalize to 0-1 range
            valid_mask = scores > float("-inf")
            if valid_mask.sum() > 1:
                min_score = scores[valid_mask].min()
                max_score = scores[valid_mask].max()
                if max_score > min_score:
                    scores[valid_mask] = (
                        (scores[valid_mask] - min_score) / (max_score - min_score)
                    )
                else:
                    scores[valid_mask] = 1.0
            scores[~valid_mask] = 0.0

            X_list.append(features)
            y_list.append(scores)

        if not X_list:
            raise ValueError("No valid examples generated from results")

        feature_names = self._get_feature_names()
        X = np.array(X_list)
        y_rankings = np.array(y_list)

        return X, y_rankings, feature_names, all_optimizers
