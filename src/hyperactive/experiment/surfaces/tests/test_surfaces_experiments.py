"""Tests for Surfaces integration experiments."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest


class TestSurfacesMathematicalExperiments:
    """Tests for mathematical test function wrappers."""

    def test_gramacy_lee_evaluation(self):
        """Test GramacyLee function evaluation."""
        from hyperactive.experiment.surfaces import GramacyLee

        func = GramacyLee()

        # Test at x=1.0 (known value)
        loss, metadata = func.evaluate({"x0": 1.0})
        assert isinstance(loss, float)
        assert isinstance(metadata, dict)
        # At x=1, f(x) = sin(10*pi)/2 + 0 ≈ 0
        assert abs(loss) < 1e-10

    def test_gramacy_lee_paramnames(self):
        """Test GramacyLee returns correct parameter names."""
        from hyperactive.experiment.surfaces import GramacyLee

        func = GramacyLee()
        assert func.paramnames() == ["x0"]

    def test_ackley2d_optimum(self):
        """Test Ackley2D at global optimum."""
        from hyperactive.experiment.surfaces import Ackley2D

        func = Ackley2D()

        # Global optimum at (0, 0) with value 0
        loss, _ = func.evaluate({"x0": 0.0, "x1": 0.0})
        assert abs(loss) < 1e-10

    def test_ackley2d_paramnames(self):
        """Test Ackley2D returns correct parameter names."""
        from hyperactive.experiment.surfaces import Ackley2D

        func = Ackley2D()
        assert func.paramnames() == ["x0", "x1"]

    def test_ackley2d_custom_params(self):
        """Test Ackley2D with custom A parameter."""
        from hyperactive.experiment.surfaces import Ackley2D

        func = Ackley2D(A=10)
        assert func.A == 10

        loss, _ = func.evaluate({"x0": 0.0, "x1": 0.0})
        assert abs(loss) < 1e-10

    def test_rastrigin_optimum(self):
        """Test Rastrigin at global optimum."""
        from hyperactive.experiment.surfaces import Rastrigin

        func = Rastrigin(n_dim=5)

        # Global optimum at origin with value 0
        params = {f"x{i}": 0.0 for i in range(5)}
        loss, _ = func.evaluate(params)
        assert abs(loss) < 1e-10

    def test_rastrigin_paramnames(self):
        """Test Rastrigin returns correct parameter names for various dimensions."""
        from hyperactive.experiment.surfaces import Rastrigin

        for n_dim in [2, 5, 10]:
            func = Rastrigin(n_dim=n_dim)
            expected = [f"x{i}" for i in range(n_dim)]
            assert func.paramnames() == expected

    def test_rastrigin_nonzero_value(self):
        """Test Rastrigin returns positive loss away from optimum."""
        from hyperactive.experiment.surfaces import Rastrigin

        func = Rastrigin(n_dim=3)

        # Away from optimum, loss should be positive
        params = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        loss, _ = func.evaluate(params)
        assert loss > 0


class TestSurfacesMLExperiments:
    """Tests for machine learning test function wrappers."""

    def test_knn_classifier_evaluation(self):
        """Test KNeighborsClassifier function evaluation."""
        from hyperactive.experiment.surfaces import KNeighborsClassifier

        func = KNeighborsClassifier(metric="accuracy")
        space = func.get_default_search_space()

        params = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "cv": 2,
            "dataset": space["dataset"][2],  # iris (fastest)
        }

        score, metadata = func.score(params)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # accuracy between 0 and 1

    def test_knn_classifier_paramnames(self):
        """Test KNeighborsClassifier returns correct parameter names."""
        from hyperactive.experiment.surfaces import KNeighborsClassifier

        func = KNeighborsClassifier()
        paramnames = func.paramnames()
        assert "n_neighbors" in paramnames
        assert "algorithm" in paramnames
        assert "cv" in paramnames
        assert "dataset" in paramnames

    def test_knn_classifier_search_space(self):
        """Test KNeighborsClassifier returns valid search space."""
        from hyperactive.experiment.surfaces import KNeighborsClassifier

        func = KNeighborsClassifier()
        space = func.get_default_search_space()

        assert "n_neighbors" in space
        assert "algorithm" in space
        assert "cv" in space
        assert "dataset" in space

        # Check types
        assert isinstance(space["n_neighbors"], list)
        assert isinstance(space["algorithm"], list)
        # n_neighbors can be numpy int64 or Python int
        assert all(isinstance(n, (int, np.integer)) for n in space["n_neighbors"])


class TestSurfacesTags:
    """Tests for tag system on Surfaces experiments."""

    def test_mathematical_function_tags(self):
        """Test that mathematical functions have correct tags."""
        from hyperactive.experiment.surfaces import Ackley2D, GramacyLee, Rastrigin

        for func_cls in [GramacyLee, Ackley2D, Rastrigin]:
            func = func_cls() if func_cls != Rastrigin else func_cls(n_dim=2)
            tags = func.get_tags()

            assert tags["property:function_family"] == "surfaces"
            assert tags["property:domain"] == "mathematical"
            assert tags["property:higher_or_lower_is_better"] == "lower"
            assert tags["property:randomness"] == "deterministic"

    def test_ml_function_tags(self):
        """Test that ML functions have correct tags."""
        from hyperactive.experiment.surfaces import KNeighborsClassifier

        func = KNeighborsClassifier()
        tags = func.get_tags()

        assert tags["property:function_family"] == "surfaces"
        assert tags["property:domain"] == "machine_learning"
        assert tags["property:higher_or_lower_is_better"] == "higher"
        assert tags["property:randomness"] == "random"

    def test_dimensionality_tags(self):
        """Test dimensionality tags are set correctly."""
        from hyperactive.experiment.surfaces import Ackley2D, GramacyLee, Rastrigin

        assert GramacyLee().get_tags()["property:dimensionality"] == "1d"
        assert Ackley2D().get_tags()["property:dimensionality"] == "2d"
        assert Rastrigin(n_dim=5).get_tags()["property:dimensionality"] == "nd"


class TestSurfacesOptimization:
    """Integration tests for optimization with Surfaces experiments."""

    def test_rastrigin_optimization(self):
        """Test optimization on Rastrigin function."""
        from hyperactive.experiment.surfaces import Rastrigin
        from hyperactive.opt.gfo import HillClimbing

        func = Rastrigin(n_dim=2)
        search_space = {
            "x0": np.linspace(-5, 5, 50),
            "x1": np.linspace(-5, 5, 50),
        }

        optimizer = HillClimbing(
            search_space=search_space,
            n_iter=30,
            experiment=func,
            random_state=42,
        )
        best_params = optimizer.solve()

        assert "x0" in best_params
        assert "x1" in best_params

        # Check that optimization found a reasonable solution
        loss, _ = func.evaluate(best_params)
        assert loss < 50  # Should be better than random

    def test_ackley2d_optimization(self):
        """Test optimization on Ackley2D function."""
        from hyperactive.experiment.surfaces import Ackley2D
        from hyperactive.opt.gfo import HillClimbing

        func = Ackley2D()
        search_space = {
            "x0": np.linspace(-5, 5, 50),
            "x1": np.linspace(-5, 5, 50),
        }

        optimizer = HillClimbing(
            search_space=search_space,
            n_iter=30,
            experiment=func,
            random_state=42,
        )
        best_params = optimizer.solve()

        assert "x0" in best_params
        assert "x1" in best_params

        # Check that optimization found a reasonable solution
        loss, _ = func.evaluate(best_params)
        assert loss < 15  # Should be better than random


class TestSurfacesScoreDirection:
    """Tests for correct score/evaluate direction handling."""

    def test_mathematical_lower_is_better(self):
        """Test that mathematical functions correctly handle lower-is-better."""
        from hyperactive.experiment.surfaces import Rastrigin

        func = Rastrigin(n_dim=2)

        # At optimum
        eval_at_opt, _ = func.evaluate({"x0": 0.0, "x1": 0.0})
        score_at_opt, _ = func.score({"x0": 0.0, "x1": 0.0})

        # Away from optimum
        eval_away, _ = func.evaluate({"x0": 2.0, "x1": 2.0})
        score_away, _ = func.score({"x0": 2.0, "x1": 2.0})

        # evaluate returns loss (lower at optimum)
        assert eval_at_opt < eval_away

        # score returns negated loss (higher at optimum)
        assert score_at_opt > score_away

        # score should be negated evaluate for lower-is-better
        assert abs(score_at_opt + eval_at_opt) < 1e-10

    def test_ml_higher_is_better(self):
        """Test that ML functions correctly handle higher-is-better."""
        from hyperactive.experiment.surfaces import KNeighborsClassifier

        func = KNeighborsClassifier()
        space = func.get_default_search_space()

        params = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "cv": 2,
            "dataset": space["dataset"][2],  # iris
        }

        eval_result, _ = func.evaluate(params)
        score_result, _ = func.score(params)

        # For ML functions (higher-is-better), evaluate and score should be equal
        assert eval_result == score_result
