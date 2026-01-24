"""Integration tests for GFO search_space formats."""

import pytest


def _make_experiment():
    from hyperactive.experiment.func import FunctionExperiment

    def objective(params):
        return params["x"] + params["y"]

    return FunctionExperiment(objective)


def _make_optimizer(search_space, experiment):
    from hyperactive.opt import GridSearch

    return GridSearch(
        search_space=search_space,
        n_iter=5,
        experiment=experiment,
    )


def test_gfo_search_space_list_coercion():
    experiment = _make_experiment()
    search_space = {"x": [1], "y": [2]}
    optimizer = _make_optimizer(search_space, experiment)
    best_params = optimizer.solve()
    assert isinstance(best_params, dict)
    assert "x" in best_params
    assert "y" in best_params
    assert best_params["x"] == 1
    assert best_params["y"] == 2


def test_gfo_search_space_union_grids_list():
    experiment = _make_experiment()
    search_space = [
        {"x": [0], "y": [0]},
        {"x": [1], "y": [2]},
    ]
    optimizer = _make_optimizer(search_space, experiment)
    best_params = optimizer.solve()
    assert isinstance(best_params, dict)
    assert "x" in best_params
    assert "y" in best_params
    assert best_params["x"] == 1
    assert best_params["y"] == 2


def test_gfo_search_space_param_grid():
    sklearn = pytest.importorskip("sklearn.model_selection")
    ParameterGrid = sklearn.ParameterGrid

    experiment = _make_experiment()
    search_space = ParameterGrid([{"x": [1], "y": [2]}])
    optimizer = _make_optimizer(search_space, experiment)
    best_params = optimizer.solve()
    assert isinstance(best_params, dict)
    assert "x" in best_params
    assert "y" in best_params
    assert best_params["x"] == 1
    assert best_params["y"] == 2
