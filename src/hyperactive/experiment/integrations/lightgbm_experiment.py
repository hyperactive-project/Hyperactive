"""Experiment adapter for LightGBM cross-validation experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.integrations.sklearn_cv import SklearnCvExperiment


class LightGBMExperiment(SklearnCvExperiment):
    """Experiment adapter for LightGBM cross-validation experiments.

    Thin wrapper around ``SklearnCvExperiment`` for LightGBM estimators.
    LightGBM's sklearn-compatible API (``LGBMClassifier``, ``LGBMRegressor``)
    works without adaptation. This class exists for discoverability, explicit
    soft-dependency tracking via the ``python_dependencies`` tag, and as an
    extension point for future LightGBM-specific behavior.

    Parameters
    ----------
    estimator : LGBMClassifier or LGBMRegressor
        The LightGBM estimator to evaluate. Any sklearn-compatible estimator
        is accepted, but LightGBM estimators are the intended use case.
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    scoring : callable or str, default=None
        Scoring function. Defaults follow ``SklearnCvExperiment`` conventions:
        ``accuracy_score`` for classifiers, ``mean_squared_error`` for
        regressors.
    cv : int or cross-validation generator, default=KFold(n_splits=3, shuffle=True)
        Cross-validation strategy.

    Notes
    -----
    LightGBM prints training logs to stdout by default. Pass
    ``verbosity=-1`` to the estimator constructor to suppress this output.

    For all remaining parameter details see ``SklearnCvExperiment``.

    Examples
    --------
    >>> from hyperactive.experiment.integrations import LightGBMExperiment
    >>> from lightgbm import LGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> exp = LightGBMExperiment(
    ...     estimator=LGBMClassifier(verbosity=-1),
    ...     X=X,
    ...     y=y,
    ... )
    >>> params = {"n_estimators": 50, "max_depth": 3}
    >>> score, metadata = exp.score(params)
    """

    _tags = {
        "authors": ["kajal-jotwani"],
        "python_dependencies": "lightgbm",
    }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skbase.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies("lightgbm", severity="none"):
            return []

        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.datasets import load_diabetes, load_iris

        X, y = load_iris(return_X_y=True)
        params0 = {
            "estimator": LGBMClassifier(n_estimators=10, verbosity=-1),
            "X": X,
            "y": y,
            "cv": 2,
        }

        X, y = load_diabetes(return_X_y=True)
        params1 = {
            "estimator": LGBMRegressor(n_estimators=10, verbosity=-1),
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

        score_params = {"n_estimators": 5, "max_depth": 2}
        return [score_params, score_params]
