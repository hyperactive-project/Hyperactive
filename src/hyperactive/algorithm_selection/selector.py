"""Algorithm selector for automatic optimizer selection.

This module provides the AlgorithmSelector class that recommends
optimization algorithms based on problem characteristics.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from typing import Callable, Optional, Type

from hyperactive.base import BaseOptimizer

try:
    from meta_learn.auto_opt.features import (
        ASTFeatures,
        SearchSpaceFeatures,
        extract_ast_features,
        extract_search_space_features,
    )
except ImportError:
    # Fall back to local copy if meta_learn is not installed
    from .ast_feature_engineering import (
        ASTFeatures,
        SearchSpaceFeatures,
        extract_ast_features,
        extract_search_space_features,
    )


class AlgorithmSelector:
    """Select the best optimization algorithm based on problem characteristics.

    This class analyzes an objective function and search space to recommend
    which optimization algorithm is likely to perform best. It returns a
    dictionary mapping optimizer classes to scores.

    The selector can use either:
    - A pre-trained ML model (when available and use_ml_model=True)
    - Heuristic rules (fallback or when use_ml_model=False)

    Parameters
    ----------
    expand_source : bool, default=True
        Whether to expand function source code to follow imports.
    use_ml_model : bool, default=True
        Whether to use the pre-trained ML model for scoring.
        Falls back to heuristics if model is not available.

    Attributes
    ----------
    ast_features_ : ASTFeatures
        AST features extracted from the last rank() call.
    search_space_features_ : SearchSpaceFeatures
        Search space features from the last rank() call.
    used_ml_model_ : bool
        Whether ML model was used in the last rank() call.

    Examples
    --------
    >>> from hyperactive.algorithm_selection import AlgorithmSelector
    >>> def objective(x):
    ...     return x["a"] ** 2 + x["b"] ** 2
    >>> search_space = {"a": list(range(-5, 5)), "b": list(range(-5, 5))}
    >>> selector = AlgorithmSelector()
    >>> rankings = selector.rank(objective, search_space)
    >>> # rankings is a dict: {OptimizerClass: score, ...}
    """

    def __init__(self, expand_source: bool = True, use_ml_model: bool = True):
        self.expand_source = expand_source
        self.use_ml_model = use_ml_model

        # Attributes set after rank() call
        self.ast_features_: Optional[ASTFeatures] = None
        self.search_space_features_: Optional[SearchSpaceFeatures] = None
        self.used_ml_model_: bool = False

    def rank(
        self,
        objective: Callable,
        search_space: dict[str, list],
        n_iter: int = 100,
    ) -> dict[Type[BaseOptimizer], float]:
        """Rank optimization algorithms for a given problem.

        Parameters
        ----------
        objective : Callable
            The objective function to optimize.
        search_space : dict
            Search space dictionary mapping parameter names to lists of values.
        n_iter : int, default=100
            Number of iterations planned for optimization.
            Used to adjust recommendations.

        Returns
        -------
        dict
            Dictionary mapping optimizer classes to scores (0.0 to 1.0).
            Higher scores indicate better expected performance.
        """
        # Extract features
        self.ast_features_ = extract_ast_features(
            objective, expand_source=self.expand_source
        )
        self.search_space_features_ = extract_search_space_features(search_space)

        # Get available optimizers
        optimizers = self._get_available_optimizers()

        # Try ML model first if enabled
        self.used_ml_model_ = False
        if self.use_ml_model:
            scores = self._score_with_ml_model(
                self.ast_features_,
                self.search_space_features_,
                n_iter,
            )
            if scores is not None:
                self.used_ml_model_ = True

        # Fall back to heuristics
        if not self.used_ml_model_:
            scores = {}
            for opt_class in optimizers:
                score = self._score_optimizer(
                    opt_class,
                    self.ast_features_,
                    self.search_space_features_,
                    n_iter,
                )
                scores[opt_class] = score

            # Normalize scores to 0-1 range
            if scores:
                max_score = max(scores.values())
                min_score = min(scores.values())
                range_score = max_score - min_score
                if range_score > 0:
                    scores = {
                        k: (v - min_score) / range_score for k, v in scores.items()
                    }

        # Sort by score descending
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return scores

    def recommend(
        self,
        objective: Callable,
        search_space: dict[str, list],
        n_iter: int = 100,
    ) -> Type[BaseOptimizer]:
        """Get the single best optimizer recommendation.

        Parameters
        ----------
        objective : Callable
            The objective function to optimize.
        search_space : dict
            Search space dictionary.
        n_iter : int, default=100
            Number of iterations planned.

        Returns
        -------
        type
            The recommended optimizer class.
        """
        rankings = self.rank(objective, search_space, n_iter)
        if not rankings:
            # Fallback to a robust default
            from hyperactive.opt import RandomSearch

            return RandomSearch
        return next(iter(rankings.keys()))

    def _get_available_optimizers(self) -> list[Type[BaseOptimizer]]:
        """Get list of available optimizer classes.

        Returns
        -------
        list of type
            Available optimizer classes from GFO backend.
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

    def _score_with_ml_model(
        self,
        ast_features: ASTFeatures,
        ss_features: SearchSpaceFeatures,
        n_iter: int,
    ) -> Optional[dict[Type[BaseOptimizer], float]]:
        """Score optimizers using the trained ML model.

        Parameters
        ----------
        ast_features : ASTFeatures
            Features extracted from objective function.
        ss_features : SearchSpaceFeatures
            Features extracted from search space.
        n_iter : int
            Planned number of iterations.

        Returns
        -------
        dict or None
            Dictionary mapping optimizer classes to scores, or None
            if the model is not available.
        """
        try:
            from .meta_learning.ranking_model import OptimizerRankingModel

            if not OptimizerRankingModel.is_model_available():
                return None

            # Pass search space features as dict for proper feature alignment
            ss_features_dict = ss_features.to_dict()

            # Pass AST features as dict
            ast_features_dict = ast_features.to_dict()

            # Get rankings from ML model (returns optimizer names)
            name_scores = OptimizerRankingModel.rank_optimizers(
                problem_features=[],  # Not used when dicts provided
                ss_features_dict=ss_features_dict,
                ast_features_dict=ast_features_dict,
                n_iter=n_iter,
            )

            if not name_scores:
                return None

            # Map names back to optimizer classes
            optimizers = self._get_available_optimizers()
            name_to_class = {opt.__name__: opt for opt in optimizers}

            scores = {}
            for name, score in name_scores.items():
                if name in name_to_class:
                    scores[name_to_class[name]] = score

            return scores if scores else None

        except Exception:
            # Any error loading/using the model falls back to heuristics
            return None

    def _score_optimizer(
        self,
        opt_class: Type[BaseOptimizer],
        ast_features: ASTFeatures,
        ss_features: SearchSpaceFeatures,
        n_iter: int,
    ) -> float:
        """Score an optimizer for the given problem characteristics.

        This is a heuristic scoring function for MVP.
        Future versions will use a trained ML model.

        Parameters
        ----------
        opt_class : type
            Optimizer class to score.
        ast_features : ASTFeatures
            Features extracted from objective function.
        ss_features : SearchSpaceFeatures
            Features extracted from search space.
        n_iter : int
            Planned number of iterations.

        Returns
        -------
        float
            Score for this optimizer (higher is better).
        """
        score = 50.0  # Base score

        # Get optimizer characteristics from tags
        try:
            opt_instance = opt_class.__new__(opt_class)
            local_vs_global = opt_class._tags.get("info:local_vs_global", "mixed")
            explore_vs_exploit = opt_class._tags.get("info:explore_vs_exploit", "mixed")
            compute = opt_class._tags.get("info:compute", "middle")
        except Exception:
            local_vs_global = "mixed"
            explore_vs_exploit = "mixed"
            compute = "middle"

        # Heuristic rules based on problem characteristics

        # Rule 1: High dimensionality favors global search
        if ss_features.n_dimensions > 10:
            if local_vs_global == "global":
                score += 15
            elif local_vs_global == "local":
                score -= 10

        # Rule 2: Small search space favors exploitation
        if ss_features.total_size < 1000:
            if explore_vs_exploit == "exploit":
                score += 10
            if compute == "low":
                score += 5

        # Rule 3: Large search space favors exploration
        if ss_features.total_size > 100000:
            if explore_vs_exploit == "explore":
                score += 15
            if local_vs_global == "global":
                score += 10

        # Rule 4: Complex objective (many operations) might be multimodal
        total_math_ops = (
            ast_features.num_add
            + ast_features.num_sub
            + ast_features.num_mult
            + ast_features.num_pow
        )
        if total_math_ops > 10:
            if local_vs_global == "global":
                score += 10
            if explore_vs_exploit == "explore":
                score += 5

        # Rule 5: Trigonometric functions suggest multimodal landscape
        trig_count = ast_features.num_sin + ast_features.num_cos + ast_features.num_tan
        if trig_count > 0:
            if local_vs_global == "global":
                score += 20
            elif local_vs_global == "local":
                score -= 15

        # Rule 6: Low iteration budget favors fast algorithms
        if n_iter < 50:
            if compute == "low":
                score += 15
            elif compute == "high":
                score -= 10

        # Rule 7: High iteration budget allows complex algorithms
        if n_iter > 500:
            if compute == "high":
                score += 10
            # Bayesian methods benefit from more iterations
            opt_name = opt_class.__name__
            if "Bayesian" in opt_name or "Forest" in opt_name:
                score += 15

        # Rule 8: Mixed parameter types favor certain algorithms
        if ss_features.has_mixed_types:
            # Some algorithms handle mixed types better
            opt_name = opt_class.__name__
            if opt_name in ("RandomSearch", "GeneticAlgorithm", "EvolutionStrategy"):
                score += 10

        # Rule 9: Many conditionals suggest discontinuous landscape
        if ast_features.num_if > 3:
            if local_vs_global == "global":
                score += 10
            if explore_vs_exploit == "explore":
                score += 5

        # Rule 10: Simple quadratic-like functions favor local search
        is_simple_quadratic = (
            ast_features.num_pow > 0
            and ast_features.num_pow <= 3
            and trig_count == 0
            and ast_features.num_if == 0
            and ss_features.n_dimensions <= 5
        )
        if is_simple_quadratic:
            if local_vs_global == "local":
                score += 20
            if explore_vs_exploit == "exploit":
                score += 10

        return score
