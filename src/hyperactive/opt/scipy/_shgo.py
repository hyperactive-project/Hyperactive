"""SHGO (Simplicial Homology Global Optimization) from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipySHGO"]


class ScipySHGO(_BaseScipyAdapter):
    """Scipy SHGO (Simplicial Homology Global Optimization).

    SHGO is designed to find all local minima of a function, not just
    the global minimum. It is effective for:

    * Problems where finding multiple local minima is valuable
    * Continuous optimization with bounds
    * Low to moderate dimensional problems

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Number of sampling iterations.

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration (not used by SHGO).

    random_state : int, optional
        Random seed (not directly supported by SHGO).

    n : int, default=100
        Number of sampling points per iteration.

    sampling_method : str, default="simplicial"
        Sampling method for generating points:

        * ``"simplicial"``: Sobol sequence based (default)
        * ``"halton"``: Halton sequence
        * ``"sobol"``: Pure Sobol sequence

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
    ScipyDirect : Another deterministic global optimizer.
    ScipyDifferentialEvolution : Stochastic global optimizer.

    References
    ----------
    .. [1] Endres, S. C., Sandrock, C., & Focke, W. W. (2018). A simplicial
           homology algorithm for Lipschitz optimisation. Journal of Global
           Optimization, 72(2), 181-217.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipySHGO

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipySHGO(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=3,
    ...     n=50,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy SHGO",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "middle",
        "python_dependencies": ["scipy"],
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        n=100,
        sampling_method="simplicial",
        experiment=None,
    ):
        self.n = n
        self.sampling_method = sampling_method

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the shgo function.

        Returns
        -------
        callable
            The ``scipy.optimize.shgo`` function.
        """
        from scipy.optimize import shgo

        return shgo

    def _get_iteration_param_name(self):
        """Get iteration parameter name.

        Returns
        -------
        str
            "iters" for shgo.
        """
        return "iters"

    def _get_optimizer_kwargs(self):
        """Get SHGO specific arguments.

        Returns
        -------
        dict
            Configuration arguments for shgo.
        """
        kwargs = {
            "n": self.n,
            "sampling_method": self.sampling_method,
        }
        return kwargs

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the SHGO optimization.

        Overrides base class to handle SHGO's different API
        (no seed, no callback).

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Number of sampling iterations.
        max_time : float, optional
            Maximum time in seconds (not supported by SHGO).
        **kwargs
            Additional parameters.

        Returns
        -------
        dict
            Best parameters found.
        """
        from scipy.optimize import shgo

        # Convert search space
        bounds, param_names = self._convert_to_scipy_space(param_space)

        # Create objective function (negated for minimization)
        def objective(x):
            params = self._array_to_dict(x, param_names)
            score = experiment(params)
            return -score

        # Run optimization
        result = shgo(
            objective,
            bounds,
            n=self.n,
            iters=n_iter,
            sampling_method=self.sampling_method,
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

        # Test 1: Default configuration
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 2,
                "n": 30,
                "experiment": ackley_exp,
            }
        )

        # Test 2: Halton sampling
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 2,
                "n": 30,
                "sampling_method": "halton",
                "experiment": ackley_exp,
            }
        )

        # Test 3: More sampling points
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 3,
                "n": 50,
                "experiment": ackley_exp,
            }
        )

        return params
