"""CMA-ES - Covariance Matrix Adaptation Evolution Strategy from Nevergrad."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters._base_nevergrad_adapter import _BaseNevergradAdapter

__all__ = ["NevergradCMA"]


class NevergradCMA(_BaseNevergradAdapter):
    """Nevergrad CMA-ES - Covariance Matrix Adaptation Evolution Strategy.

    CMA-ES is a powerful evolutionary algorithm for continuous optimization.
    It's particularly effective for:

    * Continuous optimization problems (not discrete/categorical)
    * Non-separable problems where variables are correlated
    * Moderate to large budgets (typically > 100 * dimension)
    * Neurocontrol and neural network weight optimization
    * Low to moderate noise environments

    CMA-ES adapts a multivariate normal distribution to the landscape,
    learning correlations between variables over time.

    Parameters
    ----------
    param_space : dict[str, tuple | list]
        The search space to explore. Dictionary with parameter names as keys.
        Values can be:

        * Tuple ``(low, high)``: Continuous range (recommended for CMA)
        * List of values: Discrete choices (CMA works best with continuous)

    n_iter : int, default=100
        Number of optimization iterations (budget).

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations

    random_state : int, optional
        Random seed for reproducibility.

    scale : float, default=1.0
        Initial step size (sigma). Controls initial exploration range.

        * Higher values: Broader initial search
        * Lower values: More local initial search

    popsize : int, optional
        Population size. If None, uses default ``4 + 3 * ln(dimension)``.
        Larger populations improve robustness but increase cost per iteration.

    diagonal : bool, default=False
        If True, use diagonal CMA (faster but ignores correlations).
        Use for high-dimensional problems where full covariance is expensive.

    elitist : bool, default=False
        If True, always keep the best solution in the population.
        Can help convergence but may reduce exploration.

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
    NevergradDE : Differential evolution, good for discrete/mixed parameters.
    NevergradOnePlusOne : Simpler optimizer for small budgets.

    Notes
    -----
    CMA-ES works best with continuous parameters. For problems with many
    categorical or discrete parameters, consider using ``NevergradDE`` or
    ``NevergradNGOpt`` instead.

    References
    ----------
    .. [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized
           self-adaptation in evolution strategies. Evolutionary computation,
           9(2), 159-195.

    .. [2] Nevergrad documentation: https://facebookresearch.github.io/nevergrad/

    Examples
    --------
    Basic usage with a benchmark function:

    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.nevergrad import NevergradCMA

    Create a benchmark experiment:

    >>> ackley = Ackley.create_test_instance()

    Configure the optimizer:

    >>> optimizer = NevergradCMA(
    ...     param_space={
    ...         "x0": (-5.0, 5.0),
    ...         "x1": (-5.0, 5.0),
    ...     },
    ...     n_iter=200,
    ...     scale=1.0,
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
    >>> optimizer = NevergradCMA(
    ...     param_space={
    ...         "C": (0.01, 100.0),
    ...         "gamma": (0.0001, 1.0),
    ...     },
    ...     n_iter=100,
    ...     experiment=sklearn_exp,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP

    Using diagonal CMA for high-dimensional problems:

    >>> optimizer_diag = NevergradCMA(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=200,
    ...     diagonal=True,  # Faster for high dimensions
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer_diag.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Nevergrad CMA-ES",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "exploit",
        "info:compute": "middle",
        "python_dependencies": ["nevergrad"],
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        scale=1.0,
        popsize=None,
        diagonal=False,
        elitist=False,
        experiment=None,
    ):
        self.scale = scale
        self.popsize = popsize
        self.diagonal = diagonal
        self.elitist = elitist

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_nevergrad_class(self):
        """Get the CMA optimizer class/factory.

        Returns the pre-configured CMA if no configuration is needed,
        or the ParametrizedCMA class if custom configuration is set.

        Returns
        -------
        class or configured instance
            Either ``ng.optimizers.CMA`` (pre-configured) or
            ``ParametrizedCMA`` class for custom configuration.
        """
        config = self._get_config_kwargs()

        if config:
            # Need custom configuration - use the class directly
            from nevergrad.optimization import optimizerlib

            return optimizerlib.ParametrizedCMA
        else:
            # Use pre-configured CMA
            import nevergrad as ng

            return ng.optimizers.CMA

    def _get_config_kwargs(self):
        """Get CMA configuration arguments.

        Returns
        -------
        dict
            Configuration arguments for ParametrizedCMA.
            Empty dict if no custom configuration is needed.
        """
        kwargs = {}

        # Only include non-default values
        if self.scale != 1.0:
            kwargs["scale"] = self.scale

        if self.popsize is not None:
            kwargs["popsize"] = self.popsize

        if self.diagonal:
            kwargs["diagonal"] = self.diagonal

        if self.elitist:
            kwargs["elitist"] = self.elitist

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

        Examples
        --------
        >>> params = NevergradCMA.get_test_params()
        >>> len(params) >= 3
        True
        """
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        params = []

        # Test 1: Basic continuous optimization
        ackley_exp = Ackley.create_test_instance()
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 30,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 2: Custom scale (sigma) and diagonal CMA
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 30,
                "scale": 0.5,
                "diagonal": True,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Elitist CMA for faster convergence
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 40,
                "elitist": True,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        # Test 4: sklearn experiment with continuous hyperparameters
        X, y = load_iris(return_X_y=True)
        svm_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
        params.append(
            {
                "param_space": {
                    "C": (0.01, 10.0),
                    "gamma": (0.0001, 1.0),
                },
                "n_iter": 25,
                "experiment": svm_exp,
                "random_state": 42,
            }
        )

        # Test 5: Large scale with custom popsize
        params.append(
            {
                "param_space": {
                    "x0": (-10.0, 10.0),
                    "x1": (-10.0, 10.0),
                },
                "n_iter": 50,
                "scale": 2.0,
                "popsize": 10,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        return params
