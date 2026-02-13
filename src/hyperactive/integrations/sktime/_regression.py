# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("sktime", severity="none"):
    from sktime.regression._delegate import _DelegatedRegressor
else:
    from skbase.base import BaseEstimator as _DelegatedRegressor

from hyperactive.experiment.integrations.sktime_regression import (
    SktimeRegressionExperiment,
)


class TSROptCV(_DelegatedRegressor):
    """Tune an sktime regressor via any optimizer in the hyperactive toolbox.

    ``TSROptCV`` uses any available tuning engine from ``hyperactive``
    to tune a regressor by backtesting.

    It passes backtesting results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``TSRGridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``TSROptCV`` does the following:

    In ``fit``:

    * wraps the ``estimator``, ``scoring``, and other parameters
      into a ``SktimeRegressionExperiment`` instance, which is passed to the
      optimizer ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_estimator_`` attributes.
    *  If ``refit=True``, ``best_estimator_`` is fitted to the entire ``y`` and ``X``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_estimator_`` if ``refit=True``.

    Parameters
    ----------
    estimator : sktime regressor, BaseRegressor instance or interface compatible
        The regressor to tune, must implement the sktime regressor interface.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    cv : int, sklearn cross-validation generator or an iterable, default=3-fold CV
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None = default = ``KFold(n_splits=3, shuffle=True)``
        - integer, number of folds folds in a ``KFold`` splitter, ``shuffle=True``
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.
        These splitters are instantiated with ``shuffle=False`` so the splits
        will be the same across calls.

    scoring : str, callable, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set. Can be:

        - a single string resolvable to an sklearn scorer
        - a callable that returns a single value;
        - ``None`` = default = ``mean_squared_error``

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    backend : string, by default "None".
        Parallelization backend to use for runs.
        Runs parallel evaluate if specified and ``strategy="refit"``.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute``, e.g., ``scheduler``.
        - "dask_lazy": any valid keys for ``dask.compute``, e.g., ``scheduler``.
        - "ray": any valid keys for ``ray.init``, e.g., ``num_cpus``.
    """

    _tags = {
        "authors": ["Omswastik-11", "fkiraly"],
        "maintainers": ["fkiraly", "SimonBlanke", "Omswastik-11"],
        "python_dependencies": "sktime",
    }

    _delegate_name = "best_estimator_"

    def __init__(
        self,
        estimator,
        optimizer,
        cv=None,
        scoring=None,
        refit=True,
        error_score=np.nan,
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the estimator to the training data.

        Parameters
        ----------
        X : sktime-compatible panel data (Panel scitype)
            Panel data container. Supported formats include:

            - ``pd.DataFrame`` with MultiIndex [instance, time] and variable columns
            - 3D ``np.array`` with shape ``[n_instances, n_dimensions, series_length]``
            - Other formats listed in ``datatypes.SCITYPE_REGISTER``

        y : sktime-compatible tabular data (Table scitype)
            Target variable, typically a 1D ``np.ndarray`` or ``pd.Series``
            of shape ``[n_instances]``.

        Returns
        -------
        self : object
            Returns self.
        """
        self.best_estimator_ = self.estimator.clone()

        experiment = SktimeRegressionExperiment(
            estimator=self.estimator,
            X=X,
            y=y,
            cv=self.cv,
            scoring=self.scoring,
            error_score=self.error_score,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        self.best_params_ = optimizer.solve()

        self.best_score_ = getattr(optimizer, "best_score_", np.nan)

        self.best_estimator_.set_params(**self.best_params_)

        if self.refit:
            self.best_estimator_.fit(X, y)

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import KFold
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
        from sktime.regression.dummy import DummyRegressor

        from hyperactive.opt.gfo import HillClimbing
        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        params_gridsearch = {
            "estimator": DummyRegressor(),
            "optimizer": GridSearchSk(param_grid={"strategy": ["mean", "median"]}),
        }
        params_randomsearch = {
            "estimator": DummyRegressor(),
            "cv": 2,
            "optimizer": RandomSearchSk(
                param_distributions={"strategy": ["mean", "median"]},
            ),
            "scoring": mean_squared_error,
        }
        params_hillclimb = {
            "estimator": KNeighborsTimeSeriesRegressor(),
            "cv": KFold(n_splits=2, shuffle=False),
            "optimizer": HillClimbing(
                search_space={"n_neighbors": [1, 2, 4]},
                n_iter=10,
                n_neighbours=5,
            ),
            "scoring": "mean_squared_error",
        }
        return [params_gridsearch, params_randomsearch, params_hillclimb]
