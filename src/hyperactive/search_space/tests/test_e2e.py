"""End-to-end tests for SearchSpace with optimizers."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.search_space import SearchSpace


class TestE2EWithGFO:
    """End-to-end tests with GFO optimizers."""

    def test_simple_optimization(self):
        """Test SearchSpace works with HillClimbing optimizer."""
        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            x=np.arange(-5, 5, 0.1),
            y=np.arange(-5, 5, 0.1),
        )

        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = HillClimbing(
            search_space=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result
        # Should find something near (0, 0)
        assert abs(result["x"]) < 3
        assert abs(result["y"]) < 3

    def test_with_categorical(self):
        """Test SearchSpace with categorical dimensions."""
        from hyperactive.opt import RandomSearch

        space = SearchSpace(
            x=np.arange(-5, 5, 0.5),
            mode=["add", "multiply"],
        )

        def objective(params):
            x = params["x"]
            if params["mode"] == "add":
                return -abs(x - 2)
            else:
                return -abs(x * 0.5)

        optimizer = RandomSearch(
            search_space=space,
            n_iter=50,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "mode" in result
        assert result["mode"] in ["add", "multiply"]

    def test_with_log_scale(self):
        """Test SearchSpace with log-scale dimensions."""
        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            lr=(1e-5, 1e-1, "log"),
            batch_size=[16, 32, 64, 128],
        )

        def objective(params):
            # Optimal around lr=0.001
            return -abs(np.log10(params["lr"]) + 3)

        optimizer = HillClimbing(
            search_space=space,
            n_iter=50,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "lr" in result
        assert 1e-5 <= result["lr"] <= 1e-1

    def test_with_constraints(self):
        """Test SearchSpace with constraints in GFO."""
        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            x=np.arange(0, 10, 0.5),
            y=np.arange(0, 10, 0.5),
        )
        space.add_constraint(lambda p: p["x"] + p["y"] < 10)

        def objective(params):
            return params["x"] + params["y"]

        optimizer = HillClimbing(
            search_space=space,
            n_iter=50,
            experiment=objective,
        )

        result = optimizer.solve()
        assert result["x"] + result["y"] < 10  # Constraint satisfied


class TestE2EWithSklearn:
    """End-to-end tests with sklearn optimizers."""

    def test_random_search(self):
        """Test SearchSpace works with RandomSearchSk."""
        from hyperactive.opt import RandomSearchSk

        space = SearchSpace(
            x=np.linspace(-5, 5, 50),
            y=np.linspace(-5, 5, 50),
        )

        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = RandomSearchSk(
            param_distributions=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result

    def test_with_scipy_distribution(self):
        """Test SearchSpace with scipy distributions."""
        import scipy.stats as st

        from hyperactive.opt import RandomSearchSk

        space = SearchSpace(
            x=st.uniform(-5, 10),  # uniform(-5, 5)
            y=st.norm(0, 2),  # normal(0, 2)
        )

        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = RandomSearchSk(
            param_distributions=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result


try:
    import optuna  # noqa: F401

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not available")
class TestE2EWithOptuna:
    """End-to-end tests with Optuna optimizers."""

    def test_tpe_optimizer(self):
        """Test SearchSpace works with TPEOptimizer."""
        from hyperactive.opt import TPEOptimizer

        space = SearchSpace(
            x=(0.0, 10.0),
            y=["a", "b", "c"],
        )

        def objective(params):
            score = -params["x"] ** 2
            if params["y"] == "a":
                score += 1
            return score

        optimizer = TPEOptimizer(
            param_space=space,
            n_trials=20,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result
        assert result["y"] in ["a", "b", "c"]

    def test_with_log_scale(self):
        """Test SearchSpace with log-scale in Optuna."""
        from hyperactive.opt import TPEOptimizer

        space = SearchSpace(
            lr=(1e-5, 1e-1, "log"),
            hidden=[32, 64, 128, 256],
        )

        def objective(params):
            return -abs(np.log10(params["lr"]) + 3)

        optimizer = TPEOptimizer(
            param_space=space,
            n_trials=20,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "lr" in result
        assert 1e-5 <= result["lr"] <= 1e-1


class TestBackwardCompatibility:
    """Test backward compatibility with dict-based search spaces."""

    def test_gfo_with_dict(self):
        """Test GFO still works with plain dict."""
        from hyperactive.opt import HillClimbing

        search_space = {
            "x": np.arange(-5, 5, 0.5),
            "y": np.arange(-5, 5, 0.5),
        }

        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = HillClimbing(
            search_space=search_space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result

    def test_sklearn_with_dict(self):
        """Test sklearn still works with plain dict."""
        from hyperactive.opt import RandomSearchSk

        param_distributions = {
            "x": np.linspace(-5, 5, 50).tolist(),
            "y": np.linspace(-5, 5, 50).tolist(),
        }

        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = RandomSearchSk(
            param_distributions=param_distributions,
            n_iter=20,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "x" in result
        assert "y" in result
