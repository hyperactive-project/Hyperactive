"""Basin-hopping optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import time

import numpy as np

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyBasinhopping"]


class ScipyBasinhopping(_BaseScipyAdapter):
    """Scipy Basin-hopping optimizer.

    Basin-hopping is a global optimization algorithm that combines random
    perturbations with local minimization. It is effective for:

    * Finding global minima in multimodal landscapes
    * Problems where local optimization is efficient
    * Continuous optimization problems

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Number of basin-hopping iterations (hops).

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations (uses first point as x0)

    random_state : int, optional
        Random seed for reproducibility.

    minimizer_method : str, default="Nelder-Mead"
        Local minimization method. Derivative-free options:

        * ``"Nelder-Mead"``: Simplex algorithm (default, recommended)
        * ``"Powell"``: Direction set method
        * ``"L-BFGS-B"``: Limited-memory BFGS with bounds
        * ``"COBYLA"``: Constrained optimization

    T : float, default=1.0
        Temperature for the Metropolis acceptance criterion.
        Higher values increase acceptance of worse solutions.

    stepsize : float, default=0.5
        Initial step size for random perturbations.

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
    ScipyDualAnnealing : Simulated annealing approach.
    ScipyDifferentialEvolution : Population-based optimizer.

    References
    ----------
    .. [1] Wales, D. J., & Doye, J. P. K. (1997). Global optimization by
           basin-hopping and the lowest energy structures of Lennard-Jones
           clusters. The Journal of Physical Chemistry A, 101(28), 5111-5116.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyBasinhopping

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipyBasinhopping(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=50,
    ...     minimizer_method="Nelder-Mead",
    ...     random_state=42,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy Basin-hopping",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
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
        minimizer_method="Nelder-Mead",
        T=1.0,
        stepsize=0.5,
        experiment=None,
    ):
        self.minimizer_method = minimizer_method
        self.T = T
        self.stepsize = stepsize

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the basinhopping function.

        Returns
        -------
        callable
            The ``scipy.optimize.basinhopping`` function.
        """
        from scipy.optimize import basinhopping

        return basinhopping

    def _get_iteration_param_name(self):
        """Get iteration parameter name.

        Returns
        -------
        str
            "niter" for basinhopping.
        """
        return "niter"

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the basin-hopping optimization.

        Overrides base class to handle basinhopping's different API.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Number of basin-hopping iterations.
        max_time : float, optional
            Maximum time in seconds.
        **kwargs
            Additional parameters.

        Returns
        -------
        dict
            Best parameters found.
        """
        from scipy.optimize import basinhopping

        # Convert search space
        bounds, param_names = self._convert_to_scipy_space(param_space)

        # Create objective function (negated for minimization)
        def objective(x):
            params = self._array_to_dict(x, param_names)
            score = experiment(params)
            return -score

        # Get initial point
        x0 = self._get_x0_from_initialize(bounds, param_names)
        if x0 is None:
            # Random initial point within bounds
            rng = np.random.RandomState(self.random_state)
            x0 = np.array([rng.uniform(low, high) for low, high in bounds])

        # Set up minimizer kwargs with bounds
        minimizer_kwargs = {
            "method": self.minimizer_method,
            "bounds": bounds,
        }

        # Set up callback for time limit
        start_time = time.time()

        def callback(x, f, accept):
            if max_time is not None:
                return time.time() - start_time > max_time
            return False

        # Run optimization
        result = basinhopping(
            objective,
            x0,
            niter=n_iter,
            T=self.T,
            stepsize=self.stepsize,
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,
            seed=self.random_state,
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

        # Test 1: Default configuration (Nelder-Mead)
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 10,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 2: Powell minimizer
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 10,
                "minimizer_method": "Powell",
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Higher temperature
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 15,
                "T": 2.0,
                "stepsize": 0.8,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        return params
