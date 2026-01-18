"""Tests for SearchSpaceAdapter encoding/decoding."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest

from hyperactive.opt._adapters._search_space_adapter import SearchSpaceAdapter


class TestSearchSpaceAdapter:
    """Tests for SearchSpaceAdapter encoding/decoding."""

    def test_encode_categorical_to_integers(self):
        """Categorical strings are encoded to integer indices."""
        space = {"kernel": ["rbf", "linear", "poly"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1, 2]}
        assert adapter.categorical_mapping == {
            "kernel": {0: "rbf", 1: "linear", 2: "poly"}
        }

    def test_decode_integers_to_categorical(self):
        """Integer indices are decoded back to original strings."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": 1})

        assert decoded == {"kernel": "linear"}

    def test_no_encoding_when_supported(self):
        """No encoding when backend supports categorical."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": True})

        assert adapter.needs_encoding is False
        assert adapter.encode() is space  # Same object, not copied

    def test_mixed_dimensions(self):
        """Categorical and numeric dimensions coexist."""
        space = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1], "C": [0.1, 1, 10]}
        assert "kernel" in adapter.categorical_mapping
        assert "C" not in adapter.categorical_mapping

    def test_wrapped_experiment_decodes(self):
        """Wrapped experiment receives decoded params."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        received_params = []

        class MockExperiment:
            def score(self, params):
                received_params.append(params.copy())
                return 1.0

        wrapped = adapter.wrap_experiment(MockExperiment())
        wrapped.score({"kernel": 1})

        assert received_params[0] == {"kernel": "linear"}

    def test_numpy_float_handling(self):
        """Numpy float indices are converted correctly."""
        np = pytest.importorskip("numpy")

        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": np.float64(0.0)})

        assert decoded == {"kernel": "rbf"}

    def test_numpy_int_handling(self):
        """Numpy integer indices are converted correctly."""
        np = pytest.importorskip("numpy")

        space = {"kernel": ["rbf", "linear", "poly"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": np.int64(2)})

        assert decoded == {"kernel": "poly"}

    def test_no_encoding_for_numeric_only_space(self):
        """No encoding needed when space contains only numeric values."""
        space = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        assert adapter.needs_encoding is False
        assert adapter.encode() is space

    def test_wrapped_experiment_callable(self):
        """Wrapped experiment is callable like original."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        class MockExperiment:
            def score(self, params):
                return 1.0 if params["kernel"] == "linear" else 0.5

        wrapped = adapter.wrap_experiment(MockExperiment())

        # Call via __call__
        result = wrapped({"kernel": 1})
        assert result == 1.0

    def test_wrapped_experiment_evaluate(self):
        """Wrapped experiment.evaluate() also decodes."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        received_params = []

        class MockExperiment:
            def score(self, params):
                return 1.0

            def evaluate(self, params):
                received_params.append(params.copy())
                return {"accuracy": 0.95}

        wrapped = adapter.wrap_experiment(MockExperiment())
        wrapped.evaluate({"kernel": 0})

        assert received_params[0] == {"kernel": "rbf"}

    def test_wrapped_experiment_forwards_attributes(self):
        """Wrapped experiment forwards attribute access to original."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        class MockExperiment:
            custom_attr = "test_value"

            def score(self, params):
                return 1.0

        wrapped = adapter.wrap_experiment(MockExperiment())

        assert wrapped.custom_attr == "test_value"

    def test_decode_preserves_non_categorical_params(self):
        """Decode preserves parameters that weren't encoded."""
        space = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": 1, "C": 1, "extra": "value"})

        assert decoded == {"kernel": "linear", "C": 1, "extra": "value"}

    def test_default_capability_is_categorical_supported(self):
        """Default capability assumes categorical is supported."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={})

        assert adapter.needs_encoding is False

    def test_multiple_categorical_dimensions(self):
        """Multiple categorical dimensions are all encoded."""
        space = {
            "kernel": ["rbf", "linear"],
            "solver": ["lbfgs", "sgd", "adam"],
        }
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1], "solver": [0, 1, 2]}
        assert adapter.categorical_mapping["kernel"] == {0: "rbf", 1: "linear"}
        assert adapter.categorical_mapping["solver"] == {0: "lbfgs", 1: "sgd", 2: "adam"}

        decoded = adapter.decode({"kernel": 0, "solver": 2})
        assert decoded == {"kernel": "rbf", "solver": "adam"}


class TestCategoricalEncodingIntegration:
    """Integration tests for categorical encoding in optimizers."""

    @pytest.fixture
    def sklearn_experiment(self):
        """Create a sklearn experiment fixture for testing."""
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from hyperactive.experiment.integrations import SklearnCvExperiment

        X, y = load_iris(return_X_y=True)
        return SklearnCvExperiment(estimator=SVC(), X=X, y=y, cv=3)

    def test_cmaes_has_categorical_false_tag(self):
        """CMA-ES optimizer should have categorical capability set to False."""
        from hyperactive.opt.optuna import CmaEsOptimizer

        opt = CmaEsOptimizer.create_test_instance()
        assert opt.get_tag("capability:categorical") is False

    def test_optuna_optimizers_have_categorical_true_tag(self):
        """Optuna TPE/Random/Grid optimizers should support categorical."""
        from hyperactive.opt.optuna import GridOptimizer, RandomOptimizer, TPEOptimizer

        for OptCls in [TPEOptimizer, RandomOptimizer, GridOptimizer]:
            opt = OptCls.create_test_instance()
            assert opt.get_tag("capability:categorical") is True

    def test_gfo_optimizers_have_categorical_false_tag(self):
        """GFO optimizers should have categorical tag set to False."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch.create_test_instance()
        # GFO does not support categorical natively - adapter handles encoding
        assert opt.get_tag("capability:categorical") is False


# All GFO optimizers
_GFO_OPTIMIZERS = [
    "RandomSearch",
    "HillClimbing",
    "StochasticHillClimbing",
    "RepulsingHillClimbing",
    "SimulatedAnnealing",
    "RandomRestartHillClimbing",
    "ParallelTempering",
    "ParticleSwarmOptimizer",
    "SpiralOptimization",
    "GeneticAlgorithm",
    "EvolutionStrategy",
    "DifferentialEvolution",
    "BayesianOptimizer",
    "TreeStructuredParzenEstimators",
    "ForestOptimizer",
    "GridSearch",
    "PatternSearch",
    "DirectAlgorithm",
    "LipschitzOptimizer",
    "PowellsMethod",
    "DownhillSimplexOptimizer",
]

_OPTUNA_OPTIMIZERS = [
    "TPEOptimizer",
    "RandomOptimizer",
    "GridOptimizer",
    "GPOptimizer",
    "QMCOptimizer",
    # CmaEsOptimizer excluded - only supports continuous ranges
    # NSGAIIOptimizer, NSGAIIIOptimizer excluded - multi-objective
]

_SKLEARN_OPTIMIZERS = [
    "GridSearchSk",
    "RandomSearchSk",
]


class TestAllOptimizersWithCategoricalSearchSpace:
    """Test that all optimizers work with categorical search spaces."""

    @pytest.fixture
    def categorical_search_space(self):
        """Search space with categorical and numeric values."""
        return {
            "x": [0, 1, 2, 3, 4],
            "option": ["a", "b", "c"],
        }

    @pytest.fixture
    def function_experiment(self):
        """Simple function experiment for fast testing."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            # Simple objective: x value + bonus for option
            bonus = {"a": 0, "b": 1, "c": 2}
            return params["x"] + bonus[params["option"]]

        return FunctionExperiment(objective)

    @pytest.mark.parametrize("optimizer_name", _GFO_OPTIMIZERS)
    def test_gfo_optimizer_with_categorical(
        self, optimizer_name, categorical_search_space, function_experiment
    ):
        """GFO optimizers should handle categorical search spaces."""
        from hyperactive.opt import gfo

        OptCls = getattr(gfo, optimizer_name)

        opt = OptCls(
            search_space=categorical_search_space,
            n_iter=3,
            experiment=function_experiment,
        )
        best_params = opt.solve()

        # Verify result contains original categorical string values
        assert isinstance(best_params, dict)
        assert "x" in best_params
        assert "option" in best_params
        assert best_params["option"] in ["a", "b", "c"]
        assert isinstance(best_params["option"], str)

    @pytest.mark.parametrize("optimizer_name", _OPTUNA_OPTIMIZERS)
    def test_optuna_optimizer_with_categorical(
        self, optimizer_name, categorical_search_space, function_experiment
    ):
        """Optuna optimizers should handle categorical search spaces."""
        from hyperactive.opt import optuna

        OptCls = getattr(optuna, optimizer_name)

        opt = OptCls(
            unified_space=categorical_search_space,
            n_trials=3,
            experiment=function_experiment,
        )
        best_params = opt.solve()

        # Verify result contains original categorical string values
        assert isinstance(best_params, dict)
        assert "x" in best_params
        assert "option" in best_params
        assert best_params["option"] in ["a", "b", "c"]
        assert isinstance(best_params["option"], str)

    @pytest.mark.parametrize("optimizer_name", _SKLEARN_OPTIMIZERS)
    def test_sklearn_optimizer_with_categorical(self, optimizer_name):
        """Sklearn-based optimizers should handle categorical search spaces."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier

        from hyperactive.experiment.integrations import SklearnCvExperiment
        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        X, y = load_iris(return_X_y=True)
        exp = SklearnCvExperiment(estimator=KNeighborsClassifier(), X=X, y=y, cv=2)

        sklearn_space = {
            "n_neighbors": [1, 3, 5],
            "weights": ["uniform", "distance"],
        }

        if optimizer_name == "GridSearchSk":
            opt = GridSearchSk(unified_space=sklearn_space, experiment=exp)
        else:
            opt = RandomSearchSk(unified_space=sklearn_space, n_iter=3, experiment=exp)

        best_params = opt.solve()

        # Verify result contains original categorical string values
        assert isinstance(best_params, dict)
        assert "weights" in best_params
        assert best_params["weights"] in ["uniform", "distance"]
        assert isinstance(best_params["weights"], str)
