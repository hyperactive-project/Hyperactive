"""Differential Evolution optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyDifferentialEvolution"]


class ScipyDifferentialEvolution(_BaseScipyAdapter):
    """Scipy Differential Evolution optimizer.

    Differential Evolution is a stochastic population-based optimization
    algorithm. It is particularly effective for:

    * Global optimization over continuous spaces
    * Non-differentiable and noisy objective functions
    * Problems with many local minima
    * Parallel evaluation scenarios

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Maximum number of generations (iterations).

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations (uses first point as x0)

    random_state : int, optional
        Random seed for reproducibility.

    strategy : str, default="best1bin"
        Differential evolution strategy. Options include:

        * ``"best1bin"``: Best member with 1 difference vector, binomial crossover
        * ``"best1exp"``: Best member with 1 difference vector, exponential crossover
        * ``"rand1exp"``: Random member with 1 difference vector, exponential
        * ``"randtobest1exp"``: Random-to-best with 1 difference vector
        * ``"best2exp"``: Best with 2 difference vectors
        * ``"rand2exp"``: Random with 2 difference vectors
        * ``"currenttobest1bin"``: Current-to-best, binomial

    mutation : tuple or float, default=(0.5, 1.0)
        Mutation constant (F). If tuple (min, max), dithering is used.
        Typical range: [0.5, 2.0].

    recombination : float, default=0.7
        Crossover probability (CR). Range: [0, 1].

    popsize : int, default=15
        Population size multiplier. Total population = popsize * dimensions.

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
    ScipyDualAnnealing : Simulated annealing variant.
    ScipyBasinhopping : Global optimization with local refinement.

    References
    ----------
    .. [1] Storn, R., & Price, K. (1997). Differential evolution - a simple
           and efficient heuristic for global optimization over continuous
           spaces. Journal of global optimization, 11(4), 341-359.

    Examples
    --------
    Basic usage with a benchmark function:

    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyDifferentialEvolution

    Create a benchmark experiment:

    >>> ackley = Ackley.create_test_instance()

    Configure the optimizer:

    >>> optimizer = ScipyDifferentialEvolution(
    ...     param_space={
    ...         "x0": (-5.0, 5.0),
    ...         "x1": (-5.0, 5.0),
    ...     },
    ...     n_iter=100,
    ...     strategy="best1bin",
    ...     random_state=42,
    ...     experiment=ackley,
    ... )

    Run optimization:

    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy Differential Evolution",
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
        strategy="best1bin",
        mutation=(0.5, 1.0),
        recombination=0.7,
        popsize=15,
        experiment=None,
    ):
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.popsize = popsize

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the differential_evolution function.

        Returns
        -------
        callable
            The ``scipy.optimize.differential_evolution`` function.
        """
        from scipy.optimize import differential_evolution

        return differential_evolution

    def _get_iteration_param_name(self):
        """Get iteration parameter name.

        Returns
        -------
        str
            "maxiter" for differential_evolution.
        """
        return "maxiter"

    def _get_optimizer_kwargs(self):
        """Get differential evolution specific arguments.

        Returns
        -------
        dict
            Configuration arguments for differential_evolution.
        """
        kwargs = {
            "strategy": self.strategy,
            "mutation": self.mutation,
            "recombination": self.recombination,
            "popsize": self.popsize,
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
                "n_iter": 20,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 2: Custom strategy and mutation
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 20,
                "strategy": "rand1bin",
                "mutation": 0.8,
                "recombination": 0.9,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Larger population
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 30,
                "popsize": 20,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        return params
