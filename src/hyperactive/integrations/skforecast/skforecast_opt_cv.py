"""Skforecast integration for hyperactive."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import copy

from sklearn.base import BaseEstimator

from hyperactive.experiment.integrations.skforecast_forecasting import (
    SkforecastExperiment,
)


class SkforecastOptCV(BaseEstimator):
    """Tune a skforecast forecaster via any optimizer in the hyperactive toolbox.

    Parameters
    ----------
    forecaster : skforecast forecaster
        The forecaster to tune.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    steps : int
        Number of steps to predict.

    metric : str or callable
        Metric used to quantify the goodness of fit of the model.
        If string, it must be a metric name allowed by skforecast
        (e.g., 'mean_squared_error').
        If callable, it must take (y_true, y_pred) and return a float.

    initial_train_size : int
        Number of samples in the initial training set.

    exog : pandas Series or DataFrame, default=None
        Exogenous variable/s used in the evaluation experiment.

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

    higher_is_better : bool, default=False
        Whether higher metric values indicate better performance.
        Set to False (default) for error metrics like MSE, MAE, MAPE where
        lower values are better. Set to True for metrics like R2 where
        higher values indicate better model performance.
    """

    _tags = {
        "authors": ["Omswastik-11", "JoaquinAmatRodrigo"],
        "maintainers": ["Omswastik-11", "fkiraly", "JoaquinAmatRodrigo", "SimonBlanke"],
        "python_dependencies": "skforecast",
    }

    def __init__(
        self,
        forecaster,
        optimizer,
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
        higher_is_better=False,
    ):
        self.forecaster = forecaster
        self.optimizer = optimizer
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
        self.higher_is_better = higher_is_better

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

        from skforecast.recursive import ForecasterRecursive
        from sklearn.ensemble import RandomForestRegressor

        from hyperactive import HillClimbingOptimizer

        forecaster = ForecasterRecursive(
            regressor=RandomForestRegressor(random_state=123),
            lags=2,
        )
        optimizer = HillClimbingOptimizer()

        params = {
            "forecaster": forecaster,
            "optimizer": optimizer,
            "steps": 3,
            "metric": "mean_squared_error",
            "initial_train_size": 10,
        }
        return [params]

    def fit(self, y, exog=None):
        """Fit to training data.

        Parameters
        ----------
        y : pandas Series
            Target time series to which to fit the forecaster.
        exog : pandas Series or DataFrame, optional
            Exogenous variables.

        Returns
        -------
        self : returns an instance of self.
        """
        current_exog = exog if exog is not None else self.exog

        experiment = SkforecastExperiment(
            forecaster=self.forecaster,
            y=y,
            steps=self.steps,
            metric=self.metric,
            initial_train_size=self.initial_train_size,
            exog=current_exog,
            refit=self.refit,
            fixed_train_size=self.fixed_train_size,
            gap=self.gap,
            allow_incomplete_fold=self.allow_incomplete_fold,
            return_best=self.return_best,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress,
            higher_is_better=self.higher_is_better,
        )

        if hasattr(self.optimizer, "clone"):
            optimizer = self.optimizer.clone()
        else:
            optimizer = copy.deepcopy(self.optimizer)

        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_forecaster_ = copy.deepcopy(self.forecaster)
        self.best_forecaster_.set_params(best_params)

        # Refit model with best parameters on the whole dataset
        self.best_forecaster_.fit(y=y, exog=current_exog)

        return self

    def predict(self, steps, exog=None, **kwargs):
        """Forecast time series at future horizon.

        Parameters
        ----------
        steps : int
            Number of steps to predict.
        exog : pandas Series or DataFrame, optional
            Exogenous variables.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        """
        return self.best_forecaster_.predict(steps=steps, exog=exog, **kwargs)
