"""Powell optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyPowell"]


class ScipyPowell(_BaseScipyAdapter):
    """Scipy Powell's conjugate direction method optimizer.

    Powell's method is a derivative-free local optimization algorithm that
    searches along conjugate directions. It is effective for:

    * Local optimization and fine-tuning
    * Moderate dimensional problems
    * Problems where derivatives are unavailable
    * Faster convergence than Nelder-Mead in some cases

    Note: This is a local optimizer. For global optimization, consider
    using it with warm_start from a global optimizer's result.

    Parameters
    ----------
    param_space : dict[str, tuple]
        The search space to explore. Dictionary with parameter names as keys.
        Values must be tuples ``(low, high)`` for continuous ranges.

    n_iter : int, default=100
        Maximum number of function evaluations.

    max_time : float, optional
        Maximum optimization time in seconds.

    initialize : dict, optional
        Initialization configuration. Supports:

        * ``{"warm_start": [{"param1": val1, ...}, ...]}``: Start with
          known good configurations (uses first point as x0)

    random_state : int, optional
        Random seed for initial point generation (if no warm_start).

    xtol : float, default=1e-4
        Relative error in parameter values for convergence.

    ftol : float, default=1e-4
        Relative error in objective function for convergence.

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
    ScipyNelderMead : Another derivative-free local optimizer.
    ScipyBasinhopping : Global optimizer with local refinement.

    References
    ----------
    .. [1] Powell, M. J. D. (1964). An efficient method for finding the
           minimum of a function of several variables without calculating
           derivatives. The computer journal, 7(2), 155-162.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyPowell

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipyPowell(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=200,
    ...     random_state=42,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy Powell",
        "info:local_vs_global": "local",
        "info:explore_vs_exploit": "exploit",
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
        xtol=1e-4,
        ftol=1e-4,
        experiment=None,
    ):
        self.xtol = xtol
        self.ftol = ftol

        super().__init__(
            param_space=param_space,
            n_iter=n_iter,
            max_time=max_time,
            initialize=initialize,
            random_state=random_state,
            experiment=experiment,
        )

    def _get_scipy_func(self):
        """Get the minimize function.

        Returns
        -------
        callable
            The ``scipy.optimize.minimize`` function.
        """
        from scipy.optimize import minimize

        return minimize

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the Powell optimization.

        Overrides base class to use scipy.optimize.minimize with
        method='Powell'.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Maximum number of function evaluations.
        max_time : float, optional
            Maximum time in seconds.
        **kwargs
            Additional parameters.

        Returns
        -------
        dict
            Best parameters found.
        """
        from scipy.optimize import minimize

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

        # Set up options
        options = {
            "maxfev": n_iter,
            "xtol": self.xtol,
            "ftol": self.ftol,
        }

        # Run optimization
        result = minimize(
            objective,
            x0,
            method="Powell",
            bounds=bounds,
            options=options,
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
                "n_iter": 100,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 2: Tighter tolerances
        params.append(
            {
                "param_space": {
                    "x0": (-5.0, 5.0),
                    "x1": (-5.0, 5.0),
                },
                "n_iter": 200,
                "xtol": 1e-6,
                "ftol": 1e-6,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Different search space
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 150,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        return params
