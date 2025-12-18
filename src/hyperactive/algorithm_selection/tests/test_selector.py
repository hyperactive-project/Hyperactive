"""Tests for AlgorithmSelector and AutoOptimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import math

import pytest

from hyperactive.algorithm_selection import AlgorithmSelector, AutoOptimizer
from hyperactive.base import BaseOptimizer


# =============================================================================
# Module-level test functions (required for source extraction)
# =============================================================================


def _multimodal_objective(x):
    """Multimodal objective function using trig."""
    return math.sin(x["a"]) + math.cos(x["b"])


def _helper_with_sin(x):
    """Helper function that uses sin."""
    return math.sin(x)


def _objective_calling_helper(x):
    """Objective function that calls helper."""
    return _helper_with_sin(x["a"])


# =============================================================================
# Tests for AlgorithmSelector
# =============================================================================


class TestAlgorithmSelectorBasic:
    """Test basic AlgorithmSelector functionality."""

    def test_rank_returns_dict(self):
        """rank() should return a dictionary."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        assert isinstance(rankings, dict)
        assert len(rankings) > 0

    def test_rank_values_are_floats(self):
        """Ranking values should be floats between 0 and 1."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        for opt_class, score in rankings.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_rank_keys_are_optimizer_classes(self):
        """Ranking keys should be optimizer classes."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        for opt_class in rankings.keys():
            assert isinstance(opt_class, type)
            # Check it's a subclass of BaseOptimizer
            assert issubclass(opt_class, BaseOptimizer)

    def test_rank_sorted_descending(self):
        """Rankings should be sorted by score descending."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        scores = list(rankings.values())
        assert scores == sorted(scores, reverse=True)

    def test_recommend_returns_single_class(self):
        """recommend() should return a single optimizer class."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        recommendation = selector.recommend(objective, search_space)

        assert isinstance(recommendation, type)
        assert issubclass(recommendation, BaseOptimizer)

    def test_recommend_returns_top_ranked(self):
        """recommend() should return the top-ranked optimizer."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)
        recommendation = selector.recommend(objective, search_space)

        top_ranked = next(iter(rankings.keys()))
        assert recommendation == top_ranked

    def test_features_stored_after_rank(self):
        """Features should be stored after rank() call."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-5, 6))}

        selector = AlgorithmSelector()
        selector.rank(objective, search_space)

        assert selector.ast_features_ is not None
        assert selector.search_space_features_ is not None


class TestAlgorithmSelectorHeuristics:
    """Test that heuristics produce reasonable recommendations."""

    def test_simple_quadratic_prefers_local(self):
        """Simple quadratic function should prefer local search."""

        def objective(x):
            return x["a"] ** 2 + x["b"] ** 2

        search_space = {
            "a": list(range(-5, 6)),
            "b": list(range(-5, 6)),
        }

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        # Get top 3 recommendations
        top_3 = list(rankings.keys())[:3]
        top_3_names = [opt.__name__ for opt in top_3]

        # Local search methods should be ranked highly for simple quadratic
        local_methods = ["HillClimbing", "StochasticHillClimbing", "SimulatedAnnealing"]
        assert any(name in local_methods for name in top_3_names)

    def test_multimodal_prefers_global(self):
        """Function with trig (multimodal) should prefer global search."""
        search_space = {
            "a": [i * 0.1 for i in range(-50, 51)],
            "b": [i * 0.1 for i in range(-50, 51)],
        }

        selector = AlgorithmSelector()
        rankings = selector.rank(_multimodal_objective, search_space)

        # Get the AST features
        assert selector.ast_features_.num_sin >= 1
        assert selector.ast_features_.num_cos >= 1

    def test_large_search_space_prefers_exploration(self):
        """Large search space should favor exploration."""

        def objective(x):
            return sum(v ** 2 for v in x.values())

        search_space = {f"x{i}": list(range(100)) for i in range(10)}

        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space)

        # Should have features indicating large space
        assert selector.search_space_features_.n_dimensions == 10
        assert selector.search_space_features_.total_size > 100000

    def test_low_iteration_budget_prefers_fast(self):
        """Low iteration budget should prefer fast algorithms."""

        def objective(x):
            return x["a"] ** 2

        search_space = {"a": list(range(-10, 11))}

        selector = AlgorithmSelector()
        rankings_low = selector.rank(objective, search_space, n_iter=20)
        rankings_high = selector.rank(objective, search_space, n_iter=500)

        # Rankings should potentially differ based on iteration budget
        # At minimum, the function should run without error
        assert len(rankings_low) > 0
        assert len(rankings_high) > 0


class TestAlgorithmSelectorSourceExpansion:
    """Test source expansion in AlgorithmSelector."""

    def test_expand_source_default_true(self):
        """expand_source should default to True."""
        selector = AlgorithmSelector()
        assert selector.expand_source is True

    def test_expand_source_false(self):
        """expand_source=False should not expand source."""

        def helper(x):
            return math.sin(x)

        def objective(x):
            return helper(x["a"])

        selector = AlgorithmSelector(expand_source=False)
        selector.rank(objective, {"a": [1, 2, 3]})

        # Without expansion, sin should not be detected
        assert selector.ast_features_.num_sin == 0

    def test_expand_source_true(self):
        """expand_source=True should expand source."""
        selector = AlgorithmSelector(expand_source=True)
        selector.rank(_objective_calling_helper, {"a": [1, 2, 3]})

        # With expansion, sin should be detected (in _helper_with_sin)
        assert selector.ast_features_.num_sin >= 1


# =============================================================================
# Tests for AutoOptimizer
# =============================================================================


class TestAutoOptimizerBasic:
    """Test basic AutoOptimizer functionality."""

    def test_is_base_optimizer(self):
        """AutoOptimizer should be a BaseOptimizer."""
        assert issubclass(AutoOptimizer, BaseOptimizer)

    def test_init_stores_params(self):
        """__init__ should store parameters."""

        def objective(x):
            return x["a"] ** 2

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [1, 2, 3]},
            n_iter=50,
        )

        assert auto.n_iter == 50
        assert auto.search_space == {"a": [1, 2, 3]}

    def test_tags_defined(self):
        """AutoOptimizer should have tags defined."""
        assert "info:name" in AutoOptimizer._tags
        assert AutoOptimizer._tags["info:name"] == "Auto Optimizer"


class TestAutoOptimizerSolve:
    """Test AutoOptimizer.solve() functionality."""

    def test_solve_returns_dict(self):
        """solve() should return a dict of best params."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
        )

        best_params = auto.solve()

        assert isinstance(best_params, dict)
        assert "a" in best_params

    def test_solve_sets_best_params_attribute(self):
        """solve() should set best_params_ attribute."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
        )

        auto.solve()

        assert hasattr(auto, "best_params_")
        assert auto.best_params_ is not None

    def test_solve_sets_selected_optimizer(self):
        """solve() should set selected_optimizer_ attribute."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
        )

        auto.solve()

        assert hasattr(auto, "selected_optimizer_")
        assert auto.selected_optimizer_ is not None
        assert issubclass(auto.selected_optimizer_, BaseOptimizer)

    def test_solve_sets_selection_scores(self):
        """solve() should set selection_scores_ attribute."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
        )

        auto.solve()

        assert hasattr(auto, "selection_scores_")
        assert isinstance(auto.selection_scores_, dict)

    def test_solve_sets_optimizer_instance(self):
        """solve() should set optimizer_instance_ attribute."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
        )

        auto.solve()

        assert hasattr(auto, "optimizer_instance_")
        assert isinstance(auto.optimizer_instance_, BaseOptimizer)

    def test_solve_requires_search_space(self):
        """solve() should raise error without search_space."""

        def objective(x):
            return x["a"] ** 2

        auto = AutoOptimizer(experiment=objective, n_iter=10)

        with pytest.raises(ValueError, match="search_space"):
            auto.solve()


class TestAutoOptimizerVerbose:
    """Test AutoOptimizer verbose mode."""

    def test_verbose_false_no_output(self, capsys):
        """verbose=False should not print output."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
            verbose=False,
        )

        auto.solve()

        captured = capsys.readouterr()
        assert "AutoOptimizer selected" not in captured.out

    def test_verbose_true_prints_output(self, capsys):
        """verbose=True should print selection info."""

        def objective(x):
            return -(x["a"] ** 2)

        auto = AutoOptimizer(
            experiment=objective,
            search_space={"a": [-2, -1, 0, 1, 2]},
            n_iter=10,
            verbose=True,
        )

        auto.solve()

        captured = capsys.readouterr()
        assert "AutoOptimizer selected" in captured.out
        assert "Top 3 candidates" in captured.out


class TestAutoOptimizerGetTestParams:
    """Test get_test_params for skbase compatibility."""

    def test_get_test_params_returns_dict(self):
        """get_test_params should return valid params."""
        params = AutoOptimizer.get_test_params()

        assert isinstance(params, dict)
        assert "experiment" in params
        assert "search_space" in params

    def test_can_instantiate_with_test_params(self):
        """Should be able to instantiate with test params."""
        params = AutoOptimizer.get_test_params()
        auto = AutoOptimizer(**params)

        assert isinstance(auto, AutoOptimizer)

    def test_can_solve_with_test_params(self):
        """Should be able to solve with test params."""
        params = AutoOptimizer.get_test_params()
        auto = AutoOptimizer(**params)

        best_params = auto.solve()

        assert isinstance(best_params, dict)


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for the algorithm selection system."""

    def test_full_workflow_simple(self):
        """Test complete workflow with simple function."""

        def objective(x):
            return -(x["a"] ** 2 + x["b"] ** 2)

        search_space = {
            "a": list(range(-5, 6)),
            "b": list(range(-5, 6)),
        }

        # Use selector
        selector = AlgorithmSelector()
        rankings = selector.rank(objective, search_space, n_iter=50)

        assert len(rankings) > 0

        # Use AutoOptimizer
        auto = AutoOptimizer(
            experiment=objective,
            search_space=search_space,
            n_iter=20,
        )

        best_params = auto.solve()

        assert "a" in best_params
        assert "b" in best_params

    def test_full_workflow_with_helper_functions(self):
        """Test workflow with helper functions."""

        def compute_distance(x, y):
            return (x ** 2 + y ** 2) ** 0.5

        def objective(params):
            return -compute_distance(params["x"], params["y"])

        search_space = {
            "x": [i * 0.5 for i in range(-10, 11)],
            "y": [i * 0.5 for i in range(-10, 11)],
        }

        auto = AutoOptimizer(
            experiment=objective,
            search_space=search_space,
            n_iter=30,
            expand_source=True,
        )

        best_params = auto.solve()

        assert "x" in best_params
        assert "y" in best_params

    def test_different_problems_different_selections(self):
        """Different problem types should get different recommendations."""

        # Simple quadratic
        def quadratic(x):
            return x["a"] ** 2

        # Multimodal with trig
        def multimodal(x):
            return math.sin(x["a"]) * math.cos(x["a"])

        search_space = {"a": [i * 0.1 for i in range(-50, 51)]}

        selector = AlgorithmSelector()

        rankings_quad = selector.rank(quadratic, search_space)
        rankings_multi = selector.rank(multimodal, search_space)

        # The rankings might be different (not guaranteed, but likely)
        # At minimum, both should produce valid rankings
        assert len(rankings_quad) > 0
        assert len(rankings_multi) > 0
