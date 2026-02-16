"""opt_cv module for Hyperactive optimization."""

from collections.abc import Callable

from sklearn.base import BaseEstimator, clone

from hyperactive.experiment.integrations.sklearn_cv import SklearnCvExperiment
from hyperactive.integrations.sklearn.best_estimator import (
    BestEstimator as _BestEstimator_,
)
from hyperactive.integrations.sklearn.checks import Checks

from ._compat import _check_method_params, _safe_refit, _safe_validate_X_y


class OptCV(BaseEstimator, _BestEstimator_, Checks):
    """Tuning an sklearn estimator via any optimizer in the hyperactive toolbox.

    ``OptCV`` uses any available tuning engine from ``hyperactive``
    to tune an sklearn estimator via cross-validation.

    It passes cross-validation results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``GridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``OptCV`` does the following:

    In ``fit``:

    * wraps the ``estimator``, ``scoring``, and other parameters
      into a ``SklearnCvExperiment`` instance, which is passed to the optimizer
      ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_estimator_`` attributes.
    * If ``refit=True``, ``best_estimator_`` is fitted to the entire ``X`` and ``y``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_estimator_`` if ``refit=True``.

    Parameters
    ----------
    estimator : sklearn BaseEstimator
        The estimator to be tuned.
    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.
    scoring : callable or str, default = accuracy_score or mean_squared_error
        sklearn scoring function or metric to evaluate the model's performance.
        Default is determined by the type of estimator:
        ``accuracy_score`` for classifiers, and
        ``mean_squared_error`` for regressors, as per sklearn convention
        through the default ``score`` method of the estimator.
    refit: bool, optional, default = True
        Whether to refit the best estimator with the entire dataset.
        If True, the best estimator is refit with the entire dataset after
        the optimization process.
        If False, does not refit, and predict is not available.
    cv : int or cross-validation generator, default = KFold(n_splits=3, shuffle=True)
        The number of folds or cross-validation strategy to be used.
        If int, the cross-validation used is KFold(n_splits=cv, shuffle=True).

    Example
    -------
    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    For illustration, we use grid search, this can be replaced by any other optimizer.

    1. defining the tuned estimator:
    >>> from sklearn.svm import SVC
    >>> from hyperactive.integrations.sklearn import OptCV
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>>
    >>> param_grid = {"kernel": ["linear", "rbf"], "C": [1, 10]}
    >>> tuned_svc = OptCV(SVC(), GridSearch(param_grid))

    2. fitting the tuned estimator:
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> tuned_svc.fit(X_train, y_train)
    OptCV(...)
    >>> y_pred = tuned_svc.predict(X_test)

    3. obtaining best parameters and best estimator
    >>> best_params = tuned_svc.best_params_
    >>> best_estimator = tuned_svc.best_estimator_
    """

    _required_parameters = ["estimator", "optimizer"]

    def __init__(
        self,
        estimator,
        optimizer,
        *,
        scoring: Callable | str | None = None,
        refit: bool = True,
        cv=None,
    ):
        super().__init__()

        self.estimator = estimator
        self.optimizer = optimizer
        self.scoring = scoring
        self.refit = refit
        self.cv = cv

    def _refit(self, X, y=None, **fit_params):
        self.best_estimator_ = clone(self.estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        self.best_estimator_.fit(X, y, **fit_params)
        return self

    def _check_data(self, X, y):
        return _safe_validate_X_y(self, X, y)

    @Checks.verify_fit
    def fit(self, X, y, **fit_params):
        """Fit the model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        X, y = self._check_data(X, y)

        fit_params = _check_method_params(X, params=fit_params)

        experiment = SklearnCvExperiment(
            estimator=self.estimator,
            scoring=self.scoring,
            cv=self.cv,
            X=X,
            y=y,
        )
        self.scorer_ = experiment.scorer_

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_estimator_ = clone(self.estimator).set_params(**best_params)

        _safe_refit(self, X, y, fit_params)

        return self

    def score(self, X, y=None, **params):
        """Return the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict
            Parameters to be passed to the underlying scorer(s).

        Returns
        -------
        score : float
            The score defined by ``scoring`` if provided, and the
            ``best_estimator_.score`` method otherwise.
        """
        return self.scorer_(self.best_estimator_, X, y, **params)

    @property
    def fit_successful(self):
        """Fit Successful function."""
        self._fit_successful
