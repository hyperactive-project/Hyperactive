"""SMAC3 Gaussian Process surrogate optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters._base_smac_adapter import _BaseSMACAdapter

__all__ = ["SmacGaussianProcess"]


class SmacGaussianProcess(_BaseSMACAdapter):
    """SMAC3 optimizer with Gaussian Process surrogate model.

    This optimizer uses SMAC3's BlackBoxFacade, which combines a Gaussian
    Process (GP) surrogate model with Expected Improvement (EI) acquisition
    function. It's particularly effective for:

    * Continuous parameter spaces (float ranges)
    * Small to moderate budgets (10-100 evaluations)
    * Low-dimensional problems (typically < 20 dimensions)
    * Problems where uncertainty estimates are valuable

    The Gaussian Process surrogate provides uncertainty estimates, which helps
    balance exploration and exploitation. However, GPs scale cubically with
    the number of observations, making this optimizer less suitable for
    large budgets.

    Parameters
    ----------
    param_space : dict[str, tuple | list]
        The search space to explore. Dictionary with parameter names as keys.
        Values can be:

        * Tuple ``(int, int)``: Integer range (e.g., ``(1, 100)``)
        * Tuple ``(float, float)``: Float range (e.g., ``(0.01, 10.0)``)
        * List of values: Categorical choices (NOT recommended for GP)

        Note: Gaussian Processes work best with continuous parameters.
        For mixed or categorical spaces, consider using ``SmacRandomForest``.

    n_iter : int, default=100
        Number of optimization iterations (trials). For GP-based optimization,
        50-100 iterations is often sufficient due to the model's sample
        efficiency.

    max_time : float, optional
        Maximum optimization time in seconds. If provided, optimization stops
        when time limit is reached even if ``n_iter`` not exhausted.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations.

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
    SmacRandomForest : Random Forest surrogate for mixed/categorical spaces.
    SmacRandomSearch : Random search baseline.

    Notes
    -----
    The Gaussian Process surrogate uses a Matern 5/2 kernel by default.
    Key characteristics:

    * Provides uncertainty estimates for exploration
    * Scales O(n^3) with number of observations
    * Does not support instance-based optimization
    * Best suited for continuous parameter spaces

    For problems with categorical parameters or large budgets, the
    ``SmacRandomForest`` optimizer is recommended.

    References
    ----------
    .. [1] Lindauer, M., et al. (2022). SMAC3: A Versatile Bayesian Optimization
           Package for Hyperparameter Optimization. JMLR.

    .. [2] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian
           Optimization of Machine Learning Algorithms. NeurIPS.

    Examples
    --------
    Basic usage with a benchmark function:

    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.smac import SmacGaussianProcess

    Create a benchmark experiment:

    >>> ackley = Ackley.create_test_instance()

    Configure the optimizer:

    >>> optimizer = SmacGaussianProcess(
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

    With scikit-learn hyperparameter optimization (continuous params only):

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC

    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

    Configure with continuous parameters:

    >>> optimizer = SmacGaussianProcess(
    ...     param_space={
    ...         "C": (0.01, 100.0),
    ...         "gamma": (0.0001, 1.0),
    ...     },
    ...     n_iter=50,  # GP is sample-efficient
    ...     experiment=sklearn_exp,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "SMAC Gaussian Process",
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
        """Get the BlackBoxFacade class.

        Returns
        -------
        class
            The SMAC BlackBoxFacade class.
        """
        from smac import BlackBoxFacade

        return BlackBoxFacade

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
        SmacGaussianProcess tests focus on continuous parameter spaces,
        where Gaussian Processes excel. Tests cover:

        * Different dimensional spaces (1D to 4D)
        * Various range sizes (narrow, wide, asymmetric)
        * Sample efficiency (fewer iterations)
        * Reproducibility with random_state
        * sklearn continuous hyperparameter optimization

        Categorical parameters are intentionally excluded since GPs
        work best with continuous spaces. Use SmacRandomForest for
        mixed or categorical parameter spaces.

        Examples
        --------
        >>> params = SmacGaussianProcess.get_test_params()
        >>> len(params) >= 1
        True
        """
        from sklearn.datasets import load_iris, load_wine
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR

        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Create Ackley instances with different dimensions
        ackley_1d = Ackley(d=1)
        ackley_2d = Ackley.create_test_instance()  # default is 2D
        ackley_3d = Ackley(d=3)
        ackley_4d = Ackley(d=4)

        X_iris, y_iris = load_iris(return_X_y=True)
        X_wine, y_wine = load_wine(return_X_y=True)

        # Test GP-1: Basic 2D continuous space
        params_2d_basic = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-2: 1D continuous space (simplest case)
        params_1d = {
            "param_space": {
                "x0": (-10.0, 10.0),
            },
            "n_iter": 15,
            "experiment": ackley_1d,
            "random_state": 42,
        }

        # Test GP-3: 3D continuous space
        params_3d = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
                "x2": (-5.0, 5.0),
            },
            "n_iter": 25,
            "experiment": ackley_3d,
            "random_state": 42,
        }

        # Test GP-4: 4D continuous space (GP limit for efficiency)
        params_4d = {
            "param_space": {
                "x0": (-3.0, 3.0),
                "x1": (-3.0, 3.0),
                "x2": (-3.0, 3.0),
                "x3": (-3.0, 3.0),
            },
            "n_iter": 30,
            "experiment": ackley_4d,
            "random_state": 42,
        }

        # Test GP-5: Narrow range (high precision optimization)
        params_narrow = {
            "param_space": {
                "x0": (-0.5, 0.5),
                "x1": (-0.5, 0.5),
            },
            "n_iter": 15,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-6: Wide range
        params_wide = {
            "param_space": {
                "x0": (-100.0, 100.0),
                "x1": (-100.0, 100.0),
            },
            "n_iter": 20,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-7: Asymmetric ranges
        params_asymmetric = {
            "param_space": {
                "x0": (-10.0, 2.0),
                "x1": (0.001, 50.0),
            },
            "n_iter": 20,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-8: Very small range (local optimization)
        params_small = {
            "param_space": {
                "x0": (-0.01, 0.01),
                "x1": (-0.01, 0.01),
            },
            "n_iter": 15,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-9: Sample efficient (fewer iterations)
        params_sample_efficient = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 10,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-10: With reproducibility test (different seeds)
        params_seed_42 = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "random_state": 42,
            "experiment": ackley_2d,
        }

        # Test GP-11: Different random state
        params_seed_123 = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "random_state": 123,
            "experiment": ackley_2d,
        }

        # Test GP-12: Sklearn SVR with continuous params only
        svr_exp = SklearnCvExperiment(estimator=SVR(), X=X_iris, y=y_iris)
        params_svr = {
            "param_space": {
                "C": (0.01, 100.0),
                "gamma": (0.0001, 1.0),
                "epsilon": (0.01, 1.0),
            },
            "n_iter": 15,
            "experiment": svr_exp,
            "random_state": 42,
        }

        # Test GP-13: Sklearn Ridge regression (simple continuous)
        ridge_exp = SklearnCvExperiment(estimator=Ridge(), X=X_wine, y=y_wine)
        params_ridge = {
            "param_space": {
                "alpha": (0.001, 100.0),
            },
            "n_iter": 15,
            "experiment": ridge_exp,
            "random_state": 42,
        }

        # Test GP-14: With warm_start
        params_warm_start = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "initialize": {"warm_start": [{"x0": 0.0, "x1": 0.0}]},
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-15: Non-deterministic setting
        params_non_det = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "deterministic": False,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        # Test GP-16: Positive-only range
        params_positive = {
            "param_space": {
                "x0": (0.1, 10.0),
                "x1": (0.1, 10.0),
            },
            "n_iter": 15,
            "experiment": ackley_2d,
            "random_state": 42,
        }

        return [
            params_2d_basic,
            params_1d,
            params_3d,
            params_4d,
            params_narrow,
            params_wide,
            params_asymmetric,
            params_small,
            params_sample_efficient,
            params_seed_42,
            params_seed_123,
            params_svr,
            params_ridge,
            params_warm_start,
            params_non_det,
            params_positive,
        ]
