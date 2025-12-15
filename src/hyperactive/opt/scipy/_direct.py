"""DIRECT (DIviding RECTangles) optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyDirect"]


class ScipyDirect(_BaseScipyAdapter):
    """Scipy DIRECT (DIviding RECTangles) optimizer.

    DIRECT is a deterministic derivative-free global optimization algorithm.
    It is effective for:

    * Problems where deterministic behavior is required
    * Lipschitz-continuous objective functions
    * Low to moderate dimensional problems
    * Finding approximate global optima efficiently

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Maximum number of function evaluations.

    max_time : float, optional
        Maximum optimization time in seconds (not supported by DIRECT).

    initialize : dict, optional
        Initialization configuration (not used by DIRECT).

    random_state : int, optional
        Random seed (not used, DIRECT is deterministic).

    eps : float, default=1e-4
        Minimal required difference of the objective function values
        between the current best and potential global minima.

    locally_biased : bool, default=True
        If True, use locally biased DIRECT (more local refinement).
        If False, use original DIRECT (more global exploration).

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
    ScipySHGO : Another deterministic global optimizer.
    ScipyDifferentialEvolution : Stochastic global optimizer.

    References
    ----------
    .. [1] Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993).
           Lipschitzian optimization without the Lipschitz constant.
           Journal of optimization Theory and Applications, 79(1), 157-181.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyDirect

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipyDirect(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=200,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy DIRECT",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "low",
        "python_dependencies": ["scipy"],
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        eps=1e-4,
        locally_biased=True,
        experiment=None,
    ):
        self.eps = eps
        self.locally_biased = locally_biased

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the direct function.

        Returns
        -------
        callable
            The ``scipy.optimize.direct`` function.
        """
        from scipy.optimize import direct

        return direct

    def _get_iteration_param_name(self):
        """Get iteration parameter name.

        Returns
        -------
        str
            "maxfun" for direct (controls function evaluations).
        """
        return "maxfun"

    def _get_optimizer_kwargs(self):
        """Get DIRECT specific arguments.

        Returns
        -------
        dict
            Configuration arguments for direct.
        """
        kwargs = {
            "eps": self.eps,
            "locally_biased": self.locally_biased,
        }
        return kwargs

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the DIRECT optimization.

        Overrides base class to handle DIRECT's different API
        (no seed, no callback, no x0).

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Maximum number of function evaluations.
        max_time : float, optional
            Maximum time (not supported by DIRECT).
        **kwargs
            Additional parameters.

        Returns
        -------
        dict
            Best parameters found.
        """
        from scipy.optimize import direct

        # Convert search space
        bounds, param_names = self._convert_to_scipy_space(param_space)

        # Create objective function (negated for minimization)
        def objective(x):
            params = self._array_to_dict(x, param_names)
            score = experiment(params)
            return -score

        # Run optimization
        result = direct(
            objective,
            bounds,
            eps=self.eps,
            maxfun=n_iter,
            locally_biased=self.locally_biased,
        )

        # Extract best parameters
        best_params = self._array_to_dict(result.x, param_names)
        self.best_score_ = -result.fun

        return best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer.

        Returns
        -------
        list of dict
            List of parameter configurations for testing.
        """
        from hyperactive.experiment.bench import Ackley

        params = []

        ackley_exp = Ackley.create_test_instance()

        # Test 1: Default configuration (locally biased)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 100,
                "experiment": ackley_exp,
            }
        )

        # Test 2: Original DIRECT (not locally biased)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 100,
                "locally_biased": False,
                "experiment": ackley_exp,
            }
        )

        # Test 3: Higher precision
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 150,
                "eps": 1e-6,
                "experiment": ackley_exp,
            }
        )

        return params
