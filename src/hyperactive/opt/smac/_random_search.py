"""SMAC3 Random Search optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters._base_smac_adapter import _BaseSMACAdapter

__all__ = ["SmacRandomSearch"]


class SmacRandomSearch(_BaseSMACAdapter):
    """SMAC3 Random Search optimizer.

    This optimizer uses SMAC3's RandomFacade, which performs pure random
    search without any surrogate model. It's useful for:

    * Baseline comparison against model-based optimizers
    * Problems where random search is competitive (high-dimensional)
    * Sanity checking optimization setups
    * Embarrassingly parallel optimization

    Random search samples configurations uniformly at random from the
    parameter space. Despite its simplicity, it can be surprisingly
    effective, especially in high-dimensional spaces where model-based
    methods struggle.

    Parameters
    ----------
    param_space : dict[str, tuple | list]
        The search space to explore. Dictionary with parameter names as keys.
        Values can be:

        * Tuple ``(int, int)``: Integer range (e.g., ``(1, 100)``)
        * Tuple ``(float, float)``: Float range (e.g., ``(0.01, 10.0)``)
        * List of values: Categorical choices (e.g., ``["rbf", "linear"]``)

    n_iter : int, default=100
        Number of random configurations to evaluate.

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known configurations before random sampling.

    random_state : int, optional
        Random seed for reproducibility.

    deterministic : bool, default=True
        Whether the objective function is deterministic.

    experiment : BaseExperiment, optional
        The experiment to optimize.

    Attributes
    ----------
    best_params_ : dict
        Best parameters found after calling ``solve()``.

    best_score_ : float
        Score of the best parameters found.

    See Also
    --------
    SmacRandomForest : Model-based optimizer with Random Forest surrogate.
    SmacGaussianProcess : Model-based optimizer with Gaussian Process surrogate.

    Notes
    -----
    Random search has several advantages:

    * No model fitting overhead
    * Trivially parallelizable
    * No risk of model misspecification
    * Works in any dimensional space

    However, it doesn't learn from previous evaluations, so it requires
    more samples than model-based methods for most problems.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter
           Optimization. JMLR.

    .. [2] Lindauer, M., et al. (2022). SMAC3: A Versatile Bayesian Optimization
           Package for Hyperparameter Optimization. JMLR.

    Examples
    --------
    Basic usage with a benchmark function:

    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.smac import SmacRandomSearch

    Create a benchmark experiment:

    >>> ackley = Ackley.create_test_instance()

    Configure the optimizer:

    >>> optimizer = SmacRandomSearch(
    ...     param_space={
    ...         "x0": (-5.0, 5.0),
    ...         "x1": (-5.0, 5.0),
    ...     },
    ...     n_iter=100,
    ...     random_state=42,
    ...     experiment=ackley,
    ... )

    Run optimization:

    >>> best_params = optimizer.solve()  # doctest: +SKIP
    >>> print(best_params)  # doctest: +SKIP
    {'x0': 0.5, 'x1': -0.3}

    Comparing random search with model-based optimization:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC

    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

    >>> random_opt = SmacRandomSearch(
    ...     param_space={
    ...         "C": (0.01, 100.0),
    ...         "gamma": (0.0001, 1.0),
    ...         "kernel": ["rbf", "linear"],
    ...     },
    ...     n_iter=100,
    ...     experiment=sklearn_exp,
    ... )
    >>> best_params = random_opt.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "SMAC Random Search",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "low",
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
        experiment=None,
    ):
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
        """Get the RandomFacade class.

        Returns
        -------
        class
            The SMAC RandomFacade class.
        """
        from smac import RandomFacade

        return RandomFacade

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
        SmacRandomSearch tests cover all parameter types since random
        search works with any search space. Tests include:

        * Continuous parameters (float ranges)
        * Integer parameters (int ranges)
        * Categorical parameters (string, numeric, boolean)
        * Mixed parameter types
        * High-dimensional spaces
        * Various iteration counts
        * Reproducibility with random_state
        * Warm start initialization

        Examples
        --------
        >>> params = SmacRandomSearch.get_test_params()
        >>> len(params) >= 1
        True
        """
        from sklearn.datasets import load_iris, load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Create Ackley instances with different dimensions
        ackley_exp = Ackley.create_test_instance()  # 2D
        ackley_8d = Ackley(d=8)

        X_iris, y_iris = load_iris(return_X_y=True)
        X_wine, y_wine = load_wine(return_X_y=True)

        # Test RS-1: Continuous parameters (float ranges)
        params_continuous = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-2: Mixed float + categorical
        params_mixed = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": [1, 2, 3, 4, 5],
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-3: Pure integer ranges
        params_integers = {
            "param_space": {
                "x0": (-10, 10),
                "x1": (0, 100),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-4: Pure categorical (string values)
        sklearn_exp_svc = SklearnCvExperiment(estimator=SVC(), X=X_iris, y=y_iris)
        params_categorical_str = {
            "param_space": {
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
            },
            "n_iter": 15,
            "experiment": sklearn_exp_svc,
            "random_state": 42,
        }

        # Test RS-5: Pure categorical (numeric values)
        params_categorical_num = {
            "param_space": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "gamma": [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
            "n_iter": 15,
            "experiment": sklearn_exp_svc,
            "random_state": 42,
        }

        # Test RS-6: Boolean categorical
        params_boolean = {
            "param_space": {
                "C": (0.1, 10.0),
                "shrinking": [True, False],
                "probability": [True, False],
            },
            "n_iter": 15,
            "experiment": sklearn_exp_svc,
            "random_state": 42,
        }

        # Test RS-7: Mixed all types (float + int + categorical)
        params_mixed_all = {
            "param_space": {
                "C": (0.01, 100.0),
                "degree": (2, 5),
                "kernel": ["rbf", "linear", "poly"],
                "shrinking": [True, False],
            },
            "n_iter": 20,
            "experiment": sklearn_exp_svc,
            "random_state": 42,
        }

        # Test RS-8: High-dimensional space (random search scales well)
        params_high_dim = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
                "x2": (-5.0, 5.0),
                "x3": (-5.0, 5.0),
                "x4": (-5.0, 5.0),
                "x5": (-5.0, 5.0),
                "x6": (-5.0, 5.0),
                "x7": (-5.0, 5.0),
            },
            "n_iter": 50,
            "experiment": ackley_8d,
            "random_state": 42,
        }

        # Test RS-9: Very high iteration count (random search is cheap)
        params_many_iter = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 100,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-10: Low iteration count
        params_few_iter = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 5,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-11: Sklearn KNN with integer params
        knn_exp = SklearnCvExperiment(
            estimator=KNeighborsClassifier(), X=X_iris, y=y_iris
        )
        params_knn = {
            "param_space": {
                "n_neighbors": (1, 20),
                "leaf_size": (10, 50),
                "p": [1, 2],
                "weights": ["uniform", "distance"],
            },
            "n_iter": 20,
            "experiment": knn_exp,
            "random_state": 42,
        }

        # Test RS-12: Sklearn RandomForest with comprehensive space
        rf_exp = SklearnCvExperiment(
            estimator=RandomForestClassifier(random_state=42),
            X=X_wine,
            y=y_wine,
            cv=3,
        )
        params_rf = {
            "param_space": {
                "n_estimators": (10, 200),
                "max_depth": (1, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
            },
            "n_iter": 30,
            "experiment": rf_exp,
            "random_state": 42,
        }

        # Test RS-13: With warm_start
        params_warm_start = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 25,
            "initialize": {"warm_start": [{"x0": 0.0, "x1": 0.0}]},
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-14: Multiple warm start points
        params_multi_warm = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 25,
            "initialize": {
                "warm_start": [
                    {"x0": 0.0, "x1": 0.0},
                    {"x0": -2.0, "x1": 2.0},
                    {"x0": 3.0, "x1": -3.0},
                ]
            },
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-15: Non-deterministic setting
        params_non_det = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "deterministic": False,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-16: Different random states for reproducibility
        params_seed_0 = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "random_state": 0,
            "experiment": ackley_exp,
        }

        # Test RS-17: Large range values
        params_large_range = {
            "param_space": {
                "x0": (-1000.0, 1000.0),
                "x1": (-1000.0, 1000.0),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test RS-18: Small range values
        params_small_range = {
            "param_space": {
                "x0": (-0.001, 0.001),
                "x1": (-0.001, 0.001),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        return [
            params_continuous,
            params_mixed,
            params_integers,
            params_categorical_str,
            params_categorical_num,
            params_boolean,
            params_mixed_all,
            params_high_dim,
            params_many_iter,
            params_few_iter,
            params_knn,
            params_rf,
            params_warm_start,
            params_multi_warm,
            params_non_det,
            params_seed_0,
            params_large_range,
            params_small_range,
        ]
