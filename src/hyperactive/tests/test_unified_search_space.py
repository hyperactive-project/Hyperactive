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
        assert opt.get_tag("capability:continuous") is True
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
