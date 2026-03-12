"""Integration test for end-to-end usage of optimizer with LightGBM experiment."""
# copyright: hyperactive developers, MIT License (see LICENSE file)


def test_endtoend_lightgbm():
    """Test end-to-end usage of HillClimbing optimizer with LightGBM experiment."""
    from skbase.utils.dependencies import _check_soft_dependencies

    if not _check_soft_dependencies("lightgbm", severity="none"):
        return None

    # define the experiment
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

    # set up the HillClimbing optimizer
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

    # run the HillClimbing optimizer
    hill_climbing.solve()

    best_params = hill_climbing.best_params_
    assert best_params is not None, "Best parameters should not be None"
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert (
        "n_estimators" in best_params
    ), "Best parameters should contain 'n_estimators'"
    assert "max_depth" in best_params, "Best parameters should contain 'max_depth'"
