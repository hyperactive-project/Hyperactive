"""Experiment adapter for LightGBM cross-validation experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.integrations.sklearn_cv import SklearnCvExperiment


class LightGBMExperiment(SklearnCvExperiment):
    """Experiment adapter for LightGBM cross-validation experiments.

    Thin wrapper around SklearnCvExperiment for LightGBM estimators.

    LightGBM estimators follow the sklearn API, so this class does not
    add new functionality beyond SklearnCvExperiment. It exists for
    discoverability and explicit LightGBM support.
    """

    _tags = {
        "python_dependencies": "lightgbm",
    }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skbase.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies("lightgbm", severity="none"):
            return []

        from sklearn.datasets import load_iris, load_diabetes
        from lightgbm import LGBMClassifier, LGBMRegressor

        # Classification test case
        X, y = load_iris(return_X_y=True)
        params0 = {
            "estimator": LGBMClassifier(n_estimators=10),
            "X": X,
            "y": y,
            "cv": 2,
        }

        # Regression test case
        X, y = load_diabetes(return_X_y=True)
        params1 = {
            "estimator": LGBMRegressor(n_estimators=10),
            "X": X,
            "y": y,
            "cv": 2,
        }

        return [params0, params1]

    @classmethod
    def _get_score_params(cls):
        """Return parameter settings for score/evaluate tests."""
        from skbase.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies("lightgbm", severity="none"):
            return []

        val0 = {"n_estimators": 5, "max_depth": 2}
        val1 = {"n_estimators": 5, "max_depth": 2}

        return [val0, val1]