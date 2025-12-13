"""SMAC3 Random Forest surrogate optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters._base_smac_adapter import _BaseSMACAdapter

__all__ = ["SmacRandomForest"]


class SmacRandomForest(_BaseSMACAdapter):
    """SMAC3 optimizer with Random Forest surrogate model.

    This optimizer uses SMAC3's HyperparameterOptimizationFacade, which
    combines a Random Forest surrogate model with Expected Improvement (EI)
    acquisition function. It's particularly effective for:

    * Mixed continuous/categorical/integer parameter spaces
    * Moderate to large budgets (50+ evaluations recommended)
    * Problems where function evaluations are expensive
    * Hyperparameter optimization of machine learning models

    The Random Forest surrogate model handles categorical parameters natively,
    making this optimizer well-suited for search spaces with parameters like
    ``kernel`` or ``activation`` that are categorical.

    Parameters
    ----------
    param_space : dict[str, tuple | list]
        The search space to explore. Dictionary with parameter names as keys.
        Values can be:

        * Tuple ``(int, int)``: Integer range (e.g., ``(1, 100)``)
        * Tuple ``(float, float)``: Float range (e.g., ``(0.01, 10.0)``)
        * List of values: Categorical choices (e.g., ``["rbf", "linear"]``)

        For ambiguous integer tuples like ``(1, 10)``, both bounds must be
        Python ``int`` type. Use ``(1.0, 10.0)`` for float ranges.

    n_iter : int, default=100
        Number of optimization iterations (trials). Each iteration evaluates
        one configuration.

    max_time : float, optional
        Maximum optimization time in seconds. If provided, optimization stops
        when time limit is reached even if ``n_iter`` not exhausted.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations to seed the optimization.

    random_state : int, optional
        Random seed for reproducibility. Controls both the surrogate model
        and the acquisition function optimizer.

    deterministic : bool, default=True
        Whether the objective function is deterministic. If False, SMAC will
        use multiple seeds per configuration to estimate variance.

    n_initial_points : int, default=10
        Number of initial random configurations before starting model-based
        optimization. More initial points improve the surrogate model quality
        but delay exploitation.

    experiment : BaseExperiment, optional
        The experiment to optimize. Can also be set via ``set_params()``.

    Attributes
    ----------
    best_params_ : dict
        Best parameters found after calling ``solve()``.

    best_score_ : float
        Score of the best parameters found.

    See Also
    --------
    SmacGaussianProcess : Gaussian Process surrogate for continuous spaces.
    SmacRandomSearch : Random search baseline using SMAC infrastructure.

    Notes
    -----
    SMAC3 (Sequential Model-based Algorithm Configuration) was developed by
    the AutoML groups at the Universities of Hannover and Freiburg.

    The optimizer internally uses:

    * Random Forest as surrogate model
    * Expected Improvement with log transformation as acquisition function
    * Sobol sequence for initial design
    * Local + random search for acquisition optimization

    References
    ----------
    .. [1] Lindauer, M., et al. (2022). SMAC3: A Versatile Bayesian Optimization
           Package for Hyperparameter Optimization. JMLR.

    .. [2] Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011). Sequential
           model-based optimization for general algorithm configuration.
           LION 5.

    Examples
    --------
    Basic usage with a benchmark function:

    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.smac import SmacRandomForest

    Create a benchmark experiment:

    >>> ackley = Ackley.create_test_instance()

    Configure the optimizer:

    >>> optimizer = SmacRandomForest(
    ...     param_space={
    ...         "x0": (-5.0, 5.0),
    ...         "x1": (-5.0, 5.0),
    ...     },
    ...     n_iter=50,
    ...     random_state=42,
    ...     experiment=ackley,
    ... )

    Run optimization:

    >>> best_params = optimizer.solve()  # doctest: +SKIP
    >>> print(best_params)  # doctest: +SKIP
    {'x0': 0.001, 'x1': -0.002}

    With scikit-learn hyperparameter optimization:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC

    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

    Configure with mixed parameter types:

    >>> optimizer = SmacRandomForest(
    ...     param_space={
    ...         "C": (0.01, 100.0),           # Float range
    ...         "gamma": (0.0001, 1.0),       # Float range
    ...         "kernel": ["rbf", "linear"],  # Categorical
    ...     },
    ...     n_iter=100,
    ...     experiment=sklearn_exp,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP

    Using warm start with known good configurations:

    >>> optimizer = SmacRandomForest(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=50,
    ...     initialize={"warm_start": [{"x0": 0.0, "x1": 0.0}]},
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "SMAC Random Forest",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "balanced",
        "info:compute": "middle",
        "python_dependencies": ["smac", "ConfigSpace"],
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        deterministic=True,
        n_initial_points=10,
        experiment=None,
    ):
        self.n_initial_points = n_initial_points

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            deterministic=deterministic,
            experiment=experiment,
        )

    def _get_facade_class(self):
        """Get the HyperparameterOptimizationFacade class.

        Returns
        -------
        class
            The SMAC HyperparameterOptimizationFacade class.
        """
        from smac import HyperparameterOptimizationFacade

        return HyperparameterOptimizationFacade

    def _get_scenario_kwargs(self):
        """Get scenario arguments.

        Returns
        -------
        dict
            Scenario arguments.
        """
        kwargs = {}
        if self.n_initial_points is not None:
            # SMAC uses this to determine initial design size
            kwargs["n_workers"] = 1  # Single worker for sequential evaluation
        return kwargs

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        list of dict
            List of parameter configurations for testing.

        Notes
        -----
        In addition to base class tests, SmacRandomForest adds:

        * Different n_initial_points values (controls exploration)
        * Mixed parameter spaces (RF handles categoricals natively)
        * Sklearn hyperparameter optimization with many param types
        * Higher dimensional spaces

        Examples
        --------
        >>> params = SmacRandomForest.get_test_params()
        >>> len(params) >= 1
        True
        """
        params = super().get_test_params(parameter_set)

        from sklearn.datasets import load_iris, load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Create Ackley instances with different dimensions
        ackley_exp = Ackley.create_test_instance()  # 2D
        ackley_3d = Ackley(d=3)
        ackley_5d = Ackley(d=5)

        X_iris, y_iris = load_iris(return_X_y=True)
        X_wine, y_wine = load_wine(return_X_y=True)

        # Test RF-1: Default n_initial_points
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 20,
                "n_initial_points": 10,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test RF-2: Small n_initial_points (more exploitation)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 15,
                "n_initial_points": 3,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test RF-3: Large n_initial_points (more exploration)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 25,
                "n_initial_points": 15,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test RF-4: Mixed params - RF handles categoricals natively
        sklearn_exp_svc = SklearnCvExperiment(estimator=SVC(), X=X_iris, y=y_iris)
        params.append(
            {
                "param_space": {
                    "C": (0.01, 100.0),
                    "gamma": (0.0001, 1.0),
                    "kernel": ["rbf", "linear", "poly"],
                    "shrinking": [True, False],
                },
                "n_iter": 15,
                "n_initial_points": 5,
                "experiment": sklearn_exp_svc,
                "random_state": 42,
            }
        )

        # Test RF-5: Comprehensive sklearn RF hyperparameter optimization
        sklearn_exp_rf = SklearnCvExperiment(
            estimator=RandomForestClassifier(random_state=42),
            X=X_wine,
            y=y_wine,
            cv=3,
        )
        params.append(
            {
                "param_space": {
                    "n_estimators": (10, 100),
                    "max_depth": (1, 15),
                    "min_samples_split": (2, 10),
                    "min_samples_leaf": (1, 5),
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False],
                },
                "n_iter": 15,
                "n_initial_points": 5,
                "experiment": sklearn_exp_rf,
                "random_state": 42,
            }
        )

        # Test RF-6: Integer + float + categorical combined (numeric categorical)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-10, 10),
                    "x2": [-3.0, -1.0, 0.0, 1.0, 3.0],
                },
                "n_iter": 15,
                "n_initial_points": 5,
                "experiment": ackley_3d,
                "random_state": 42,
            }
        )

        # Test RF-7: Higher dimensional space (RF scales well)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                    "x2": (-5.0, 5.0),
                    "x3": (-5.0, 5.0),
                    "x4": (-5.0, 5.0),
                },
                "n_iter": 20,
                "n_initial_points": 10,
                "experiment": ackley_5d,
                "random_state": 42,
            }
        )

        # Test RF-8: With warm_start and n_initial_points
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 20,
                "n_initial_points": 5,
                "initialize": {"warm_start": [{"x0": 0.0, "x1": 0.0}]},
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test RF-9: Many categorical options
        params.append(
            {
                "param_space": {
                    "C": (0.1, 10.0),
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "degree": [2, 3, 4, 5],
                    "gamma": ["scale", "auto"],
                },
                "n_iter": 15,
                "n_initial_points": 5,
                "experiment": sklearn_exp_svc,
                "random_state": 42,
            }
        )

        return params
