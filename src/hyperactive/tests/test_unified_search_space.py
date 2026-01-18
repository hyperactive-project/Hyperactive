"""Tests for unified_space parameter across all backends."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from hyperactive.experiment.integrations import SklearnCvExperiment


@pytest.fixture
def sklearn_experiment():
    """Create a sklearn experiment fixture for testing."""
    X, y = load_iris(return_X_y=True)
    return SklearnCvExperiment(estimator=SVC(), X=X, y=y, cv=3)


@pytest.fixture
def simple_search_space():
    """Simple unified search space format: dict[str, list]."""
    return {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}


class TestUnifiedSearchSpaceGFO:
    """Test unified_space parameter for GFO optimizers."""

    def test_gfo_random_search_accepts_unified_space(
        self, sklearn_experiment, simple_search_space
    ):
        """GFO RandomSearch should accept unified_space."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch(
            unified_space=simple_search_space,
            n_iter=5,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_gfo_search_space_still_works(
        self, sklearn_experiment, simple_search_space
    ):
        """Backward compatibility: search_space (native GFO) should still work."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch(
            search_space=simple_search_space,
            n_iter=5,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_gfo_raises_when_both_provided(
        self, sklearn_experiment, simple_search_space
    ):
        """GFO should raise when both unified_space and search_space are given."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch(
            unified_space=simple_search_space,
            search_space=simple_search_space,
            n_iter=5,
            experiment=sklearn_experiment,
        )

        with pytest.raises(ValueError, match="Provide either 'unified_space' or"):
            opt.solve()


class TestUnifiedSearchSpaceOptuna:
    """Test unified_space parameter for Optuna optimizers."""

    @pytest.mark.parametrize(
        "optimizer_cls",
        [
            pytest.param("TPEOptimizer", id="tpe"),
            pytest.param("RandomOptimizer", id="random"),
            pytest.param("GridOptimizer", id="grid"),
        ],
    )
    def test_optuna_accepts_unified_space(
        self, sklearn_experiment, simple_search_space, optimizer_cls
    ):
        """Optuna optimizers should accept unified_space."""
        import hyperactive.opt.optuna as optuna_module

        OptCls = getattr(optuna_module, optimizer_cls)

        opt = OptCls(
            unified_space=simple_search_space,
            n_trials=5,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_optuna_param_space_still_works(self, sklearn_experiment):
        """Backward compatibility: param_space should still work."""
        from hyperactive.opt.optuna import TPEOptimizer

        # Native Optuna format with ranges
        param_space = {"C": (0.1, 10), "gamma": [0.01, 0.1, 1]}

        opt = TPEOptimizer(
            param_space=param_space,
            n_trials=5,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_optuna_raises_when_both_provided(self, sklearn_experiment):
        """Optuna should raise when both unified_space and param_space are given."""
        from hyperactive.opt.optuna import TPEOptimizer

        unified_space = {"C": [0.1, 1], "gamma": [0.01]}
        param_space = {"C": (0.1, 10), "gamma": [0.01, 0.1, 1]}

        opt = TPEOptimizer(
            unified_space=unified_space,
            param_space=param_space,
            n_trials=5,
            experiment=sklearn_experiment,
        )

        with pytest.raises(ValueError, match="Provide either 'unified_space' or"):
            opt.solve()


class TestUnifiedSearchSpaceGridSearch:
    """Test unified_space parameter for GridSearchSk."""

    def test_gridsearch_accepts_unified_space(
        self, sklearn_experiment, simple_search_space
    ):
        """GridSearchSk should accept unified_space."""
        from hyperactive.opt import GridSearchSk

        opt = GridSearchSk(
            unified_space=simple_search_space,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_gridsearch_param_grid_still_works(self, sklearn_experiment):
        """Backward compatibility: param_grid should still work."""
        from hyperactive.opt import GridSearchSk

        param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}

        opt = GridSearchSk(
            param_grid=param_grid,
            experiment=sklearn_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params

    def test_gridsearch_raises_when_both_provided(
        self, sklearn_experiment, simple_search_space
    ):
        """GridSearchSk should raise when both unified_space and param_grid given."""
        from hyperactive.opt import GridSearchSk

        opt = GridSearchSk(
            unified_space=simple_search_space,
            param_grid=simple_search_space,
            experiment=sklearn_experiment,
        )

        with pytest.raises(ValueError, match="Provide either 'unified_space' or"):
            opt.solve()


class TestCapabilityTags:
    """Test capability tags for search space features."""

    def test_gfo_capability_tags(self):
        """GFO optimizers should have correct capability tags."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch.create_test_instance()

        assert opt.get_tag("capability:discrete") is True
        assert opt.get_tag("capability:continuous") is False  # GFO needs lists, not tuples
        assert opt.get_tag("capability:categorical") is False  # GFO only numeric
        assert opt.get_tag("capability:constraints") is True

    def test_optuna_capability_tags(self):
        """Optuna optimizers should have correct capability tags."""
        from hyperactive.opt.optuna import TPEOptimizer

        opt = TPEOptimizer.create_test_instance()

        assert opt.get_tag("capability:discrete") is True
        assert opt.get_tag("capability:continuous") is True
        assert opt.get_tag("capability:categorical") is True
        assert opt.get_tag("capability:log_scale") is True


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
        from sklearn.neighbors import KNeighborsClassifier

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


class TestAllOptimizersWithContinuousSearchSpace:
    """Test that all optimizers work with continuous search spaces (tuples)."""

    @pytest.fixture
    def continuous_search_space(self):
        """Search space with continuous dimensions (tuples)."""
        return {
            "x": (0.0, 10.0),  # linear, default 100 points
            "y": (1e-4, 1e-1, "log"),  # log scale, 100 points
        }

    @pytest.fixture
    def continuous_search_space_custom_points(self):
        """Search space with custom n_points."""
        return {
            "x": (0.0, 10.0, 20),  # linear, 20 points
            "y": (1e-4, 1e-1, 15, "log"),  # log scale, 15 points
        }

    @pytest.fixture
    def mixed_search_space(self):
        """Search space with continuous, discrete, and categorical dimensions."""
        return {
            "x": (0.0, 5.0, 10),  # continuous
            "n": [1, 2, 3, 4, 5],  # discrete numeric
            "option": ["a", "b", "c"],  # categorical
        }

    @pytest.fixture
    def continuous_function_experiment(self):
        """Simple function experiment for continuous optimization."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            # Simple objective: maximize when x and y are in middle of range
            x = params["x"]
            y = params["y"]
            return -(x - 5.0) ** 2 - (y - 0.01) ** 2

        return FunctionExperiment(objective)

    @pytest.fixture
    def mixed_function_experiment(self):
        """Function experiment for mixed search space."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            bonus = {"a": 0, "b": 1, "c": 2}
            return params["x"] + params["n"] + bonus[params["option"]]

        return FunctionExperiment(objective)

    @pytest.mark.parametrize("optimizer_name", _GFO_OPTIMIZERS)
    def test_gfo_optimizer_with_continuous(
        self, optimizer_name, continuous_search_space, continuous_function_experiment
    ):
        """GFO optimizers should handle continuous search spaces via discretization."""
        from hyperactive.opt import gfo

        OptCls = getattr(gfo, optimizer_name)

        opt = OptCls(
            search_space=continuous_search_space,
            n_iter=3,
            experiment=continuous_function_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "x" in best_params
        assert "y" in best_params
        # Values should be within the specified ranges
        assert 0.0 <= best_params["x"] <= 10.0
        assert 1e-4 <= best_params["y"] <= 1e-1
        # Values should be floats
        assert isinstance(best_params["x"], float)
        assert isinstance(best_params["y"], float)

    @pytest.mark.parametrize("optimizer_name", _GFO_OPTIMIZERS)
    def test_gfo_optimizer_with_custom_n_points(
        self,
        optimizer_name,
        continuous_search_space_custom_points,
        continuous_function_experiment,
    ):
        """GFO optimizers should handle custom n_points in continuous dimensions."""
        from hyperactive.opt import gfo

        OptCls = getattr(gfo, optimizer_name)

        opt = OptCls(
            search_space=continuous_search_space_custom_points,
            n_iter=3,
            experiment=continuous_function_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert 0.0 <= best_params["x"] <= 10.0
        assert 1e-4 <= best_params["y"] <= 1e-1

    @pytest.mark.parametrize("optimizer_name", _GFO_OPTIMIZERS)
    def test_gfo_optimizer_with_mixed_space(
        self, optimizer_name, mixed_search_space, mixed_function_experiment
    ):
        """GFO optimizers should handle mixed continuous/discrete/categorical."""
        from hyperactive.opt import gfo

        OptCls = getattr(gfo, optimizer_name)

        opt = OptCls(
            search_space=mixed_search_space,
            n_iter=3,
            experiment=mixed_function_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        # Continuous
        assert 0.0 <= best_params["x"] <= 5.0
        assert isinstance(best_params["x"], float)
        # Discrete
        assert best_params["n"] in [1, 2, 3, 4, 5]
        # Categorical (should be decoded back to string)
        assert best_params["option"] in ["a", "b", "c"]
        assert isinstance(best_params["option"], str)

    @pytest.mark.parametrize("optimizer_name", _OPTUNA_OPTIMIZERS)
    def test_optuna_optimizer_with_continuous(
        self, optimizer_name, continuous_search_space, continuous_function_experiment
    ):
        """Optuna optimizers should handle continuous search spaces natively."""
        from hyperactive.opt import optuna

        OptCls = getattr(optuna, optimizer_name)

        opt = OptCls(
            unified_space=continuous_search_space,
            n_trials=3,
            experiment=continuous_function_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "x" in best_params
        assert "y" in best_params
        assert 0.0 <= best_params["x"] <= 10.0
        assert 1e-4 <= best_params["y"] <= 1e-1

    @pytest.mark.parametrize("optimizer_name", _OPTUNA_OPTIMIZERS)
    def test_optuna_optimizer_with_mixed_space(
        self, optimizer_name, mixed_search_space, mixed_function_experiment
    ):
        """Optuna optimizers should handle mixed continuous/discrete/categorical."""
        from hyperactive.opt import optuna

        OptCls = getattr(optuna, optimizer_name)

        opt = OptCls(
            unified_space=mixed_search_space,
            n_trials=3,
            experiment=mixed_function_experiment,
        )
        best_params = opt.solve()

        assert isinstance(best_params, dict)
        # Continuous
        assert 0.0 <= best_params["x"] <= 5.0
        # Discrete
        assert best_params["n"] in [1, 2, 3, 4, 5]
        # Categorical
        assert best_params["option"] in ["a", "b", "c"]
        assert isinstance(best_params["option"], str)

    @pytest.mark.parametrize("optimizer_name", _SKLEARN_OPTIMIZERS)
    def test_sklearn_optimizer_with_continuous(self, optimizer_name):
        """Sklearn-based optimizers should handle continuous via discretization."""
        from sklearn.neighbors import KNeighborsClassifier

        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        X, y = load_iris(return_X_y=True)
        exp = SklearnCvExperiment(estimator=KNeighborsClassifier(), X=X, y=y, cv=2)

        # Continuous search space for sklearn
        sklearn_space = {
            "n_neighbors": [1, 3, 5, 7],  # discrete (sklearn needs this)
            "leaf_size": (10, 50, 5),  # continuous, 5 points
        }

        if optimizer_name == "GridSearchSk":
            opt = GridSearchSk(unified_space=sklearn_space, experiment=exp)
        else:
            opt = RandomSearchSk(unified_space=sklearn_space, n_iter=3, experiment=exp)

        best_params = opt.solve()

        assert isinstance(best_params, dict)
        assert "n_neighbors" in best_params
        assert "leaf_size" in best_params
        assert 10 <= best_params["leaf_size"] <= 50
