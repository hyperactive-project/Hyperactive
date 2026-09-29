"""Integration tests for end-to-end usage of optimizers with experiments.

API unit tests are in TestAllOptimizers and TestAllExperiments.
"""
# copyright: hyperactive developers, MIT License (see LICENSE file)


def test_endtoend_hillclimbing():
    """Test end-to-end usage of HillClimbing optimizer with an experiment."""
    # 1. define the experiment
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.svm import SVC

    from hyperactive.experiment.integrations import SklearnCvExperiment

    X, y = load_iris(return_X_y=True)

    sklearn_exp = SklearnCvExperiment(
        estimator=SVC(),
        scoring=accuracy_score,
        cv=KFold(n_splits=3, shuffle=True),
        X=X,
        y=y,
    )

    # 2. set up the HillClimbing optimizer
    import numpy as np

    from hyperactive.opt import HillClimbing

    hillclimbing_config = {
        "search_space": {
            "C": np.array([0.01, 0.1, 1, 10]),
            "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
        },
        "n_iter": 100,
    }
    hill_climbing = HillClimbing(**hillclimbing_config, experiment=sklearn_exp)

    # 3. run the HillClimbing optimizer
    hill_climbing.solve()

    best_params = hill_climbing.best_params_
    assert best_params is not None, "Best parameters should not be None"
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert "C" in best_params, "Best parameters should contain 'C'"
    assert "gamma" in best_params, "Best parameters should contain 'gamma'"


def test_endtoend_lightgbm():
    """Test end-to-end usage of HillClimbing optimizer with LightGBM experiment."""
    from skbase.utils.dependencies import _check_soft_dependencies

    if not _check_soft_dependencies("lightgbm", severity="none"):
        return None

    # 1. define the experiment
    from lightgbm import LGBMClassifier
    from sklearn.datasets import load_iris

    from hyperactive.experiment.integrations import LightGBMExperiment

    X, y = load_iris(return_X_y=True)

    lgbm_exp = LightGBMExperiment(
        estimator=LGBMClassifier(n_estimators=10, verbosity=-1),
        X=X,
        y=y,
        cv=2,
    )

    # 2. set up the HillClimbing optimizer
    import numpy as np

    from hyperactive.opt import HillClimbing

    hillclimbing_config = {
        "search_space": {
            "n_estimators": np.array([5, 10, 20]),
            "max_depth": np.array([2, 3, 5]),
        },
        "n_iter": 10,
    }
    hill_climbing = HillClimbing(**hillclimbing_config, experiment=lgbm_exp)

    # 3. run the HillClimbing optimizer
    hill_climbing.solve()

    best_params = hill_climbing.best_params_
    assert best_params is not None, "Best parameters should not be None"
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert (
        "n_estimators" in best_params
    ), "Best parameters should contain 'n_estimators'"
    assert "max_depth" in best_params, "Best parameters should contain 'max_depth'"
