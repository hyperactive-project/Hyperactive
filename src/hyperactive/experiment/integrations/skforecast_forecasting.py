"""Experiment adapter for skforecast backtesting experiments."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import copy

from hyperactive.base import BaseExperiment


class SkforecastExperiment(BaseExperiment):
    """Experiment adapter for skforecast backtesting experiments.

    This class is used to perform backtesting experiments using a given
    skforecast forecaster. It allows for hyperparameter tuning and evaluation of
    the model's performance.

    Parameters
    ----------
    forecaster : skforecast forecaster
        skforecast forecaster to benchmark.

    y : pandas Series
        Target time series used in the evaluation experiment.

    exog : pandas Series or DataFrame, default=None
        Exogenous variable/s used in the evaluation experiment.

    steps : int
        Number of steps to predict.

    metric : str or callable
        Metric used to quantify the goodness of fit of the model.
        If string, it must be a metric name allowed by skforecast
        (e.g., 'mean_squared_error').
        If callable, it must take (y_true, y_pred) and return a float.

    initial_train_size : int
        Number of samples in the initial training set.

    refit : bool, default=False
        Whether to re-fit the forecaster in each iteration.

    fixed_train_size : bool, default=False
        If True, the train size doesn't increase but moves by `steps` in each iteration.

    gap : int, default=0
        Number of samples to exclude from the end of each training set and the
        start of the test set.

    allow_incomplete_fold : bool, default=True
        If True, the last fold is allowed to have fewer samples than `steps`.

    return_best : bool, default=False
        If True, the best model is returned.

    n_jobs : int or 'auto', default="auto"
        Number of jobs to run in parallel.

    verbose : bool, default=False
        Print summary figures.

    show_progress : bool, default=False
        Whether to show a progress bar.
    """

    _tags = {
        "authors": ["Omswastik-11", "JoaquinAmatRodrigo"],
        "maintainers": ["Omswastik-11", "fkiraly", "JoaquinAmatRodrigo", "SimonBlanke"],
        "python_dependencies": "skforecast",
    }

    def __init__(
        self,
        forecaster,
        y,
        steps,
        metric,
        initial_train_size,
        exog=None,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        return_best=False,
        n_jobs="auto",
        verbose=False,
        show_progress=False,
    ):
        self.forecaster = forecaster
        self.y = y
        self.steps = steps
        self.metric = metric
        self.initial_train_size = initial_train_size
        self.exog = exog
        self.refit = refit
        self.fixed_train_size = fixed_train_size
        self.gap = gap
        self.allow_incomplete_fold = allow_incomplete_fold
        self.return_best = return_best
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress

        super().__init__()

        # All standard skforecast/sklearn regression metrics are "lower is better"
        # (MSE, MAE, MAPE, RMSE, etc.). Custom callables are assumed lower is better.
        self.set_tags(**{"property:higher_or_lower_is_better": "lower"})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance,
            i.e., MyClass(**params) or MyClass(**params[i]) creates a valid test
            instance.
            create_test_instance uses the first (or only) dictionary in `params`
        """
        from skbase.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies("skforecast", severity="none"):
            return []

        import numpy as np
        import pandas as pd
        from skforecast.recursive import ForecasterRecursive
        from sklearn.ensemble import RandomForestRegressor

        forecaster = ForecasterRecursive(
            regressor=RandomForestRegressor(random_state=123),
            lags=2,
        )

        y = pd.Series(
            np.random.randn(20),
            index=pd.date_range(start="2020-01-01", periods=20, freq="D"),
            name="y",
        )

        params = {
            "forecaster": forecaster,
            "y": y,
            "steps": 3,
            "metric": "mean_squared_error",
            "initial_train_size": 10,
        }
        return [params]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        return [{"n_estimators": 5}]

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
        from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

        forecaster = copy.deepcopy(self.forecaster)
        forecaster.set_params(params)

        cv = TimeSeriesFold(
            steps=self.steps,
            initial_train_size=self.initial_train_size,
            refit=self.refit,
            fixed_train_size=self.fixed_train_size,
            gap=self.gap,
            allow_incomplete_fold=self.allow_incomplete_fold,
        )

        results, _ = backtesting_forecaster(
            forecaster=forecaster,
            y=self.y,
            cv=cv,
            metric=self.metric,
            exog=self.exog,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )

        if isinstance(self.metric, str):
            metric_name = self.metric
        else:
            metric_name = (
                self.metric.__name__ if hasattr(self.metric, "__name__") else "score"
            )

        # backtesting_forecaster returns a DataFrame
        res_float = results[metric_name].iloc[0]

        return res_float, {"results": results}
