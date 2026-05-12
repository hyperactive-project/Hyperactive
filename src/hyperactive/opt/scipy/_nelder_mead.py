"""Nelder-Mead optimizer from scipy.optimize."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.opt._adapters import _BaseScipyAdapter

__all__ = ["ScipyNelderMead"]


class ScipyNelderMead(_BaseScipyAdapter):
    """Scipy Nelder-Mead simplex optimizer.

    Nelder-Mead is a derivative-free local optimization algorithm that uses
    a simplex to explore the search space. It is effective for:

    * Local optimization and fine-tuning
    * Low-dimensional problems (typically < 10 dimensions)
    * Smooth objective functions
    * Problems where derivatives are unavailable

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

    xatol : float, default=1e-4
        Absolute error in parameter values for convergence.

    fatol : float, default=1e-4
        Absolute error in objective function for convergence.

    adaptive : bool, default=True
        Adapt algorithm parameters to dimensionality.

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
    ScipyPowell : Another derivative-free local optimizer.
    ScipyBasinhopping : Global optimizer with local refinement.

    References
    ----------
    .. [1] Nelder, J. A., & Mead, R. (1965). A simplex method for function
           minimization. The computer journal, 7(4), 308-313.

    Examples
    --------
    >>> from hyperactive.experiment.bench import Ackley
    >>> from hyperactive.opt.scipy import ScipyNelderMead

    >>> ackley = Ackley.create_test_instance()
    >>> optimizer = ScipyNelderMead(
    ...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
    ...     n_iter=200,
    ...     random_state=42,
    ...     experiment=ackley,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "info:name": "Scipy Nelder-Mead",
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
        xatol=1e-4,
        fatol=1e-4,
        adaptive=True,
        experiment=None,
    ):
        self.xatol = xatol
        self.fatol = fatol
        self.adaptive = adaptive

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
        """Run the Nelder-Mead optimization.

        Overrides base class to use scipy.optimize.minimize with
        method='Nelder-Mead'.

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
            "xatol": self.xatol,
            "fatol": self.fatol,
            "adaptive": self.adaptive,
        }

        # Run optimization
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
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
                "xatol": 1e-6,
                "fatol": 1e-6,
                "experiment": ackley_exp,
                "random_state": 42,
            }
        )

        # Test 3: Non-adaptive
        params.append(
            {
                "param_space": {
                    "x0": (-3.0, 3.0),
                    "x1": (-3.0, 3.0),
                },
                "n_iter": 150,
                "adaptive": False,
                "experiment": ackley_exp,
                "random_state": 123,
            }
        )

        return params
