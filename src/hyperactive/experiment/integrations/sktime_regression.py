"""Experiment adapter for sktime regression experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment
from hyperactive.experiment.integrations._skl_metrics import _coerce_to_scorer_and_sign


class SktimeRegressionExperiment(BaseExperiment):
    """Experiment adapter for time series regression experiments.

    This class is used to perform cross-validation experiments using a given
    sktime regressor. It allows for hyperparameter tuning and evaluation of
    the model's performance.

    The score returned is the summary backtesting score,
    of applying ``sktime`` ``evaluate`` to ``estimator`` with the parameters given in
    ``score`` ``params``.

    The backtesting performed is specified by the ``cv`` parameter,
    and the scoring metric is specified by the ``scoring`` parameter.
    The ``X`` and ``y`` parameters are the input data and target values,
    which are used in fit/predict cross-validation.

    Parameters
    ----------
    estimator : sktime BaseRegressor descendant (concrete regressor)
        sktime regressor to benchmark

    X : sktime-compatible panel data (Panel scitype)
        Panel data container. Supported formats include:

        - ``pd.DataFrame`` with MultiIndex [instance, time] and variable columns
        - 3D ``np.array`` with shape ``[n_instances, n_dimensions, series_length]``
        - Other formats listed in ``datatypes.SCITYPE_REGISTER``

    y : sktime-compatible tabular data (Table scitype)
        Target variable, typically a 1D ``np.ndarray`` or ``pd.Series``
        of shape ``[n_instances]``.

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
        "authors": ["fkiraly", "Omswastik-11"],
        "maintainers": ["SimonBlanke", "fkiraly", "Omswastik-11"],
        "python_dependencies": "sktime",
    }

    def __init__(
        self,
        estimator,
        X,
        y,
        cv=None,
        scoring=None,
        error_score=np.nan,
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        # normalize scoring and infer optimization direction
        self._scoring, _sign = _coerce_to_scorer_and_sign(scoring, "regressor")
        higher_or_lower_better = "higher" if _sign == 1 else "lower"
        self.set_tags(**{"property:higher_or_lower_is_better": higher_or_lower_better})

        # default handling for cv
        if isinstance(cv, int):
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=cv, shuffle=True)
        elif cv is None:
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=3, shuffle=True)
        else:
            self._cv = cv

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return list(self.estimator.get_params().keys())

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        from sktime.classification.model_evaluation import evaluate

        estimator = self.estimator.clone().set_params(**params)

        # determine metric function for sktime.evaluate via centralized coerce helper
        metric_func = getattr(self._scoring, "_metric_func", None)
        if metric_func is None:
            # defensive fallback; _coerce_to_scorer should always attach one
            from sklearn.metrics import (
                mean_squared_error as metric_func,  # type: ignore
            )

        results = evaluate(
            estimator,
            cv=self._cv,
            X=self.X,
            y=self.y,
            scoring=metric_func,
            error_score=self.error_score,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        metric = metric_func
        result_name = f"test_{getattr(metric, '__name__', 'score')}"

        res_float = results[result_name].mean()

        return res_float, {"results": results}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import KFold
        from sktime.datasets import load_unit_test
        from sktime.regression.dummy import DummyRegressor

        X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")
        y = y.astype(float)

        params0 = {
            "estimator": DummyRegressor(strategy="mean"),
            "X": X,
            "y": y,
        }

        params1 = {
            "estimator": DummyRegressor(strategy="median"),
            "cv": KFold(n_splits=2),
            "X": X,
            "y": y,
            "scoring": mean_absolute_error,
        }

        def passthrough_scorer(estimator, X, y):
            return estimator.score(X, y)

        params2 = {
            "estimator": DummyRegressor(strategy="quantile", quantile=0.5),
            "X": X,
            "y": y,
            "cv": KFold(n_splits=2),
            "scoring": passthrough_scorer,
        }

        return [params0, params1, params2]

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        val0 = {}
        val1 = {"strategy": "mean"}
        return [val0, val1]
