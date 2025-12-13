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

        Examples
        --------
        >>> params = SmacRandomSearch.get_test_params()
        >>> len(params) >= 1
        True
        """
        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()

        # Test with continuous parameters
        params_continuous = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        # Test with mixed parameters
        params_mixed = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": [1, 2, 3, 4, 5],
            },
            "n_iter": 20,
            "experiment": ackley_exp,
            "random_state": 42,
        }

        return [params_continuous, params_mixed]
