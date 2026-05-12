# copyright: hyperactive developers, MIT License (see LICENSE file)

import time

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

_HAS_SKTIME = _check_soft_dependencies("sktime", severity="none")

if _HAS_SKTIME:
    from sktime.forecasting.base._delegate import _DelegatedForecaster
else:
    from skbase.base import BaseEstimator as _DelegatedForecaster

from hyperactive.experiment.integrations.sktime_forecasting import (
    SktimeForecastingExperiment,
)


class ForecastingOptCV(_DelegatedForecaster):
    """Tune an sktime forecaster via any optimizer in the hyperactive toolbox.

    ``ForecastingOptCV`` uses any available tuning engine from ``hyperactive``
    to tune a forecaster by backtesting.

    It passes backtesting results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``ForecastingGridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``ForecastingOptCV`` does the following:

    In ``fit``:

    * wraps the ``forecaster``, ``scoring``, and other parameters
      into a ``SktimeForecastingExperiment`` instance, which is passed to the optimizer
      ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_forecaster_`` attributes.
    *  If ``refit=True``, ``best_forecaster_`` is fitted to the entire ``y`` and ``X``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_forecaster_`` if ``refit=True``.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above
        if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        if ``cv_X`` is passed, ``X`` is split according to ``cv_X``

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update

    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    cv_X : sktime BaseSplitter descendant, optional
        determines split of ``X`` into test and train folds
        default is ``X`` being split to identical ``loc`` indices as ``y``
        if passed, must have same number of splits as ``cv``

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
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    tune_by_instance : bool, optional (default=False)
        Whether to tune parameters separately for each time series instance when
        panel or hierarchical data is passed. Mirrors ``ForecastingGridSearchCV``
        semantics by delegating broadcasting to sktime's vectorization logic.
    tune_by_variable : bool, optional (default=False)
        Whether to tune parameters per variable for strictly multivariate series.
        When enabled, only univariate targets are accepted and internal
        broadcasting is handled by sktime.

    Example
    -------
    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    For illustration, we use grid search, this can be replaced by any other optimizer.

    1. defining the tuned estimator:
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from hyperactive.integrations.sktime import ForecastingOptCV
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>>
    >>> param_grid = {"strategy": ["mean", "last", "drift"]}
    >>> tuned_naive = ForecastingOptCV(
    ...     NaiveForecaster(),
    ...     GridSearch(param_grid),
    ...     cv=ExpandingWindowSplitter(
    ...         initial_window=12, step_length=3, fh=range(1, 13)
    ...     ),
    ... )

    2. fitting the tuned estimator:
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>>
    >>> tuned_naive.fit(y_train, fh=range(1, 13))
    ForecastingOptCV(...)
    >>> y_pred = tuned_naive.predict()

    3. obtaining best parameters and best forecaster
    >>> best_params = tuned_naive.best_params_
    >>> best_forecaster = tuned_naive.best_forecaster_

    Attributes
    ----------
    best_params_ : dict
        Best parameter values returned by the optimizer.
    best_forecaster_ : estimator
        Fitted estimator with the best parameters.
    best_score_ : float
        Score of the best model (according to ``scoring``, after hyperactive's
        "higher-is-better" normalization).
    best_index_ : int or None
        Index of the best parameter combination if the optimizer exposes it.
    scorer_ : BaseMetric
        The scoring object resolved by ``check_scoring``.
    n_splits_ : int or None
        Number of splits produced by ``cv`` (if the splitter exposes it).
    cv_results_ : pd.DataFrame or None
        Evaluation table returned by ``sktime.evaluate`` for the winning parameters.
        (Full per-candidate traces require the optimizer to provide detailed metadata.)
    refit_time_ : float
        Time in seconds to refit the best forecaster when ``refit=True``.
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "best_forecaster_"

    def __init__(
        self,
        forecaster,
        optimizer,
        cv,
        strategy="refit",
        update_behaviour="full_refit",
        scoring=None,
        refit=True,
        error_score=np.nan,
        cv_X=None,
        backend=None,
        backend_params=None,
        tune_by_instance=False,
        tune_by_variable=False,
    ):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.cv = cv
        self.strategy = strategy
        self.update_behaviour = update_behaviour
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.cv_X = cv_X
        self.backend = backend
        self.backend_params = backend_params
        self.tune_by_instance = tune_by_instance
        self.tune_by_variable = tune_by_variable
        super().__init__()

        if _HAS_SKTIME:
            self._set_delegated_tags(delegate=self.forecaster)
            tags_to_clone = ["y_inner_mtype", "X_inner_mtype"]
            self.clone_tags(self.forecaster, tags_to_clone)
            self._extend_to_all_scitypes("y_inner_mtype")
            self._extend_to_all_scitypes("X_inner_mtype")

            if self.tune_by_variable:
                self.set_tags(**{"scitype:y": "univariate"})

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        from sktime.utils.validation.forecasting import check_scoring

        forecaster = self.forecaster.clone()

        scoring = check_scoring(self.scoring, obj=self)
        self.scorer_ = scoring
        get_n_splits = getattr(self.cv, "get_n_splits", None)
        if callable(get_n_splits):
            try:
                self.n_splits_ = get_n_splits(y)
            except TypeError:
                # fallback for splitters that expect no args
                self.n_splits_ = get_n_splits()
        else:
            self.n_splits_ = None
        # scoring_name = f"test_{scoring.name}"

        experiment = SktimeForecastingExperiment(
            forecaster=forecaster,
            scoring=scoring,
            cv=self.cv,
            X=X,
            y=y,
            strategy=self.strategy,
            error_score=self.error_score,
            cv_X=self.cv_X,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_index_ = getattr(optimizer, "best_index_", None)
        raw_best_score, best_metadata = experiment.evaluate(best_params)
        self.best_score_ = float(raw_best_score)
        results_table = best_metadata.get("results") if best_metadata else None
        if results_table is not None:
            try:
                self.cv_results_ = results_table.copy()
            except AttributeError:
                self.cv_results_ = results_table
        else:
            self.cv_results_ = None
        self.best_forecaster_ = forecaster.set_params(**best_params)

        # Refit model with best parameters.
        if self.refit:
            refit_start = time.perf_counter()
            self.best_forecaster_.fit(y=y, X=X, fh=fh)
            self.refit_time_ = time.perf_counter() - refit_start
        else:
            self.refit_time_ = 0.0

        return self

    def _extend_to_all_scitypes(self, tagname):
        """Ensure mtypes for all scitypes are present in tag ``tagname``."""
        from sktime.datatypes import mtype_to_scitype

        tagval = self.get_tag(tagname)
        if not isinstance(tagval, list):
            tagval = [tagval]
        scitypes = mtype_to_scitype(tagval, return_unique=True)

        if "Series" not in scitypes:
            tagval = tagval + ["pd.DataFrame"]
        elif "pd.Series" in tagval and "pd.DataFrame" not in tagval:
            tagval = ["pd.DataFrame"] + tagval

        if "Panel" not in scitypes:
            tagval = tagval + ["pd-multiindex"]
        if "Hierarchical" not in scitypes:
            tagval = tagval + ["pd_multiindex_hier"]

        if self.tune_by_instance:
            tagval = [x for x in tagval if mtype_to_scitype(x) == "Series"]

        self.set_tags(**{tagname: tagval})

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
        return super()._predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        update_behaviour = self.update_behaviour

        if update_behaviour == "full_refit":
            super()._update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "inner_only":
            self.best_forecaster_.update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "no_update":
            self.best_forecaster_.update(y=y, X=X, update_params=False)
        else:
            raise ValueError(
                'update_behaviour must be one of "full_refit", "inner_only",'
                f' or "no_update", but found {update_behaviour}'
            )
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
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import (
            MeanAbsolutePercentageError,
            mean_absolute_percentage_error,
        )
        from sktime.split import SingleWindowSplitter

        from hyperactive.opt.gfo import HillClimbing
        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        params_gridsearch = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": GridSearchSk(param_grid={"window_length": [2, 5]}),
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }
        params_randomsearch = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": RandomSearchSk(param_distributions={"degree": [1, 2]}),
            "scoring": mean_absolute_percentage_error,
            "update_behaviour": "inner_only",
        }
        params_hillclimb = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": HillClimbing(
                search_space={"window_length": [2, 5]},
                n_iter=10,
                n_neighbours=5,
            ),
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }
        return [params_gridsearch, params_randomsearch, params_hillclimb]
