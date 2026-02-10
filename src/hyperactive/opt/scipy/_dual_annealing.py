"""Dual Annealing optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyDualAnnealing"]


class ScipyDualAnnealing(_BaseScipyAdapter):
    """Scipy Dual Annealing optimizer.

    Dual Annealing combines Classical Simulated Annealing with a fast
    local search method. It is effective for:

    * Global optimization with many local minima
    * Continuous optimization problems
    * Problems where local refinement improves solutions

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Maximum number of global iterations.

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations (uses first point as x0)

    random_state : int, optional
        Random seed for reproducibility.

    initial_temp : float, default=5230.0
        Initial temperature for the annealing schedule.

    restart_temp_ratio : float, default=2e-5
        When temperature falls below ``initial_temp * restart_temp_ratio``,
        the annealing restarts.

    visit : float, default=2.62
        Parameter for the visiting distribution. Higher values lead to
        heavier tails (more global exploration).

    accept : float, default=-5.0
        Parameter for the acceptance distribution. More negative values
        make acceptance stricter.

    no_local_search : bool, default=False
        If True, disable local search refinement.

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
    ScipyDifferentialEvolution : Population-based global optimizer.
    ScipyBasinhopping : Another global-local hybrid approach.

    References
    ----------
    .. [1] Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs
           statistics. Journal of statistical physics, 52(1-2), 479-487.

    .. [2] Xiang, Y., et al. (2013). Generalized simulated annealing for
           global optimization. Science, 220(4598), 671-680.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyDualAnnealing

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipyDualAnnealing(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=100,
    ...     random_state=42,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy Dual Annealing",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
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
        initial_temp=5230.0,
        restart_temp_ratio=2e-5,
        visit=2.62,
        accept=-5.0,
        no_local_search=False,
        experiment=None,
    ):
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept
        self.no_local_search = no_local_search

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the dual_annealing function.

        Returns
        -------
        callable
            The ``scipy.optimize.dual_annealing`` function.
        """
        from scipy.optimize import dual_annealing

        return dual_annealing

    def _get_iteration_param_name(self):
        """Get iteration parameter name.

        Returns
        -------
        str
            "maxiter" for dual_annealing.
        """
        return "maxiter"

    def _get_optimizer_kwargs(self):
        """Get dual annealing specific arguments.

        Returns
        -------
        dict
            Configuration arguments for dual_annealing.
        """
        kwargs = {
            "initial_temp": self.initial_temp,
            "restart_temp_ratio": self.restart_temp_ratio,
            "visit": self.visit,
            "accept": self.accept,
            "no_local_search": self.no_local_search,
        }
        return kwargs

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
                "n_iter": 50,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 2: No local search
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 50,
                "no_local_search": True,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Custom temperature settings
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 30,
                "initial_temp": 10000.0,
                "visit": 2.8,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        return params
