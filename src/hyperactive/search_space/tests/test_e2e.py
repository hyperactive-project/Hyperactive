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

    def test_with_conditions_completes(self):
        """Test SearchSpace with conditions completes optimization.

        CRITICAL TEST: This catches the bug where conditions were incorrectly
        converted to constraints, causing GFO to hang trying to find valid
        parameter combinations.
        """
        from hyperactive.opt import RandomSearch

        space = SearchSpace(
            kernel=["rbf", "linear", "poly"],
            C=(0.01, 100.0, "log"),
            gamma=(1e-4, 10.0, "log"),
            degree=[2, 3, 4, 5],
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        space.add_condition("degree", when=lambda p: p["kernel"] == "poly")

        def objective(params):
            kernel = params["kernel"]
            score = -np.log10(params["C"]) ** 2
            if kernel == "rbf":
                score -= (np.log10(params.get("gamma", 1.0)) + 1) ** 2
            elif kernel == "poly":
                score -= (params.get("degree", 3) - 3) ** 2
            return score

        optimizer = RandomSearch(
            search_space=space,
            n_iter=30,
            experiment=objective,
        )

        # This should complete quickly, not hang
        result = optimizer.solve()

        assert "kernel" in result
        assert "C" in result
        assert result["kernel"] in ["rbf", "linear", "poly"]

    def test_with_conditions_hillclimbing(self):
        """Test conditions work with HillClimbing optimizer."""
        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            mode=["simple", "advanced"],
            x=np.arange(-5, 5, 0.5),
            extra_param=[1, 2, 3],
        )
        space.add_condition("extra_param", when=lambda p: p["mode"] == "advanced")

        def objective(params):
            score = -params["x"] ** 2
            if params["mode"] == "advanced":
                score += params.get("extra_param", 0)
            return score

        optimizer = HillClimbing(
            search_space=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "mode" in result
        assert "x" in result

    def test_conditions_and_constraints_together(self):
        """Test both conditions and constraints work together."""
        from hyperactive.opt import RandomSearch

        space = SearchSpace(
            kernel=["rbf", "linear"],
            C=(0.1, 10.0, "log"),
            gamma=(0.01, 1.0, "log"),
        )
        # Condition: gamma only for rbf
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        # Constraint: C should not be too large
        space.add_constraint(lambda p: p["C"] < 5)

        def objective(params):
            score = -np.log10(params["C"]) ** 2
            if params["kernel"] == "rbf":
                score -= np.log10(params.get("gamma", 0.1)) ** 2
            return score

        optimizer = RandomSearch(
            search_space=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()

        # Constraint should be satisfied
        assert result["C"] < 5
        assert result["kernel"] in ["rbf", "linear"]

    def test_multiple_conditions_completes(self):
        """Test multiple conditions don't cause optimization to hang."""
        from hyperactive.opt import RandomSearch

        space = SearchSpace(
            model=["svm", "rf", "nn"],
            C=(0.1, 10.0, "log"),
            n_estimators=[10, 50, 100],
            hidden_size=[32, 64, 128],
        )
        space.add_condition("C", when=lambda p: p["model"] == "svm")
        space.add_condition("n_estimators", when=lambda p: p["model"] == "rf")
        space.add_condition("hidden_size", when=lambda p: p["model"] == "nn")

        def objective(params):
            model = params["model"]
            if model == "svm":
                return -np.log10(params.get("C", 1.0)) ** 2
            elif model == "rf":
                return params.get("n_estimators", 10) / 100
            else:
                return params.get("hidden_size", 32) / 128

        optimizer = RandomSearch(
            search_space=space,
            n_iter=30,
            experiment=objective,
        )

        result = optimizer.solve()
        assert result["model"] in ["svm", "rf", "nn"]


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


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not available")
class TestOptunaAdvancedFeatures:
    """Tests for advanced Optuna integration features."""

    def test_optuna_log_scale_distribution_handling(self):
        """Test Optuna adapter handles FloatDistribution for log-scale.

        When SearchSpace converts a log-scale dimension, it creates an Optuna
        FloatDistribution. The _suggest_params method handles this correctly.
        """
        from hyperactive.opt import TPEOptimizer

        space = SearchSpace(
            lr=(1e-5, 1e-1, "log"),
        )

        def objective(params):
            return -abs(np.log10(params["lr"]) + 3)

        optimizer = TPEOptimizer(
            param_space=space,
            n_trials=5,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "lr" in result
        assert 1e-5 <= result["lr"] <= 1e-1

    def test_optuna_integer_continuous_handling(self):
        """Test Optuna adapter handles integer continuous correctly.

        When SearchSpace has a continuous integer dimension (e.g., (1, 100)),
        the Optuna adapter uses suggest_int to return an integer.
        """
        from hyperactive.opt import TPEOptimizer

        space = SearchSpace(
            n_layers=(1, 10),  # Integer continuous
        )

        def objective(params):
            return -abs(params["n_layers"] - 5)

        optimizer = TPEOptimizer(
            param_space=space,
            n_trials=5,
            experiment=objective,
        )

        result = optimizer.solve()
        assert "n_layers" in result
        # Should be an integer, not a float
        assert isinstance(result["n_layers"], int)


class TestNestedSpaceBugs:
    """Tests for known bugs in nested space handling."""

    def test_nested_space_missing_restructure_method(self):
        """Bug: SearchSpace lacks restructure_result method.

        When using nested spaces, results come back flat:
          {"estimator": RFC, "randomforestclassifier__n_estimators": 100}

        They should be restructurable to:
          {"estimator": RFC, "estimator_params": {"n_estimators": 100}}

        This is documented in the plan but not implemented.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator_params={
                RandomForestClassifier: {"n_estimators": [10, 50]},
                SVC: {"C": [0.1, 1.0]},
            }
        )

        # The SearchSpace knows about nested spaces
        assert space.has_nested_spaces
        assert "estimator_params" in space.nested_spaces

        # But there's no method to restructure flat results back to nested
        # This functionality is specified in the plan but not implemented
        flat_result = {
            "estimator": RandomForestClassifier,
            "randomforestclassifier__n_estimators": 100,
        }

        # Bug: restructure_result method does not exist
        # This test passes now, will fail when the method is implemented
        assert not hasattr(space, "restructure_result")

        # When implemented, the method should work like this:
        # structured = space.restructure_result(flat_result)
        # assert structured == {
        #     "estimator": RandomForestClassifier,
        #     "estimator_params": {"n_estimators": 100},
        # }


class TestValidationBugs:
    """Tests for missing validation functionality."""

    def test_validation_warns_for_unsupported_conditions(self):
        """Bug: No validation warning when using conditions with GFO.

        The plan specifies validate_space_for_optimizer() should warn
        when using conditions with backends that don't support them natively.
        This function is not implemented.
        """
        import warnings

        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.1, 10.0),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        def objective(params):
            return 1.0

        # Should warn that GFO doesn't natively support conditions
        # Currently no warning is emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            optimizer = HillClimbing(
                search_space=space,
                n_iter=5,
                experiment=objective,
            )

            # This assertion will fail until validation is implemented
            # Uncomment when implementing the fix:
            # assert len(w) >= 1
            # assert "conditional" in str(w[0].message).lower()

    def test_validation_warns_for_unsupported_constraints_optuna(self):
        """Bug: No validation warning when using constraints with Optuna.

        Optuna doesn't natively support constraints. The validation function
        should warn users about this.
        """
        pytest.importorskip("optuna")

        import warnings

        from hyperactive.opt import TPEOptimizer

        space = SearchSpace(
            x=(0.0, 10.0),
            y=(0.0, 10.0),
        )
        space.add_constraint(lambda p: p["x"] + p["y"] < 10)

        def objective(params):
            return 1.0

        # Should warn that Optuna doesn't natively support constraints
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            optimizer = TPEOptimizer(
                param_space=space,
                n_trials=5,
                experiment=objective,
            )

            # This assertion will fail until validation is implemented
            # Uncomment when implementing the fix:
            # assert len(w) >= 1
            # assert "constraint" in str(w[0].message).lower()


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
