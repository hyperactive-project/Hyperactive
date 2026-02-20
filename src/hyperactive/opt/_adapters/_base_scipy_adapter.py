"""Base adapter for scipy.optimize optimizers."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import time

import numpy as np

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseScipyAdapter"]


class _BaseScipyAdapter(BaseOptimizer):
    """Base adapter class for scipy.optimize optimizers.

    This adapter handles the conversion between Hyperactive's interface and
    scipy's optimization functions. Key responsibilities:

    * Search space conversion to scipy bounds format
    * Score negation (scipy minimizes, Hyperactive maximizes)
    * Array-to-dict parameter conversion
    * Random state handling
    * Time-based early stopping via callbacks

    Extension interface for subclasses:

    * ``_get_scipy_func``: Return the scipy optimization function
    * ``_get_optimizer_kwargs``: Return optimizer-specific kwargs
    * ``_get_iteration_param_name``: Return the parameter name for iterations

    Notes
    -----
    Scipy optimizers are designed for continuous optimization. This adapter
    only supports continuous parameter spaces (tuples). For discrete or
    categorical parameters, use the optuna or gfo backends instead.

    Scipy minimizes objectives, while Hyperactive maximizes (higher scores
    are better). This adapter negates scores when calling scipy functions.
    """

    _tags = {
        "python_dependencies": ["scipy"],
        "info:name": "Scipy-based optimizer",
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        experiment=None,
    ):
        self.param_space = param_space
        self.n_iter = n_iter
        self.max_time = max_time
        self.initialize = initialize
        self.random_state = random_state
        self.experiment = experiment
        super().__init__()

    def _get_scipy_func(self):
        """Get the scipy optimization function to use.

        Returns
        -------
        callable
            The scipy optimization function. Must be a function from
            ``scipy.optimize``.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_scipy_func to return "
            "the scipy optimization function."
        )

    def _get_optimizer_kwargs(self):
        """Get optimizer-specific keyword arguments.

        Override this method in subclasses to pass algorithm-specific
        parameters to the scipy optimization function.

        Returns
        -------
        dict
            Keyword arguments to pass to the optimizer.
            Default is an empty dict.
        """
        return {}

    def _get_iteration_param_name(self):
        """Get the parameter name used for iteration control.

        Different scipy optimizers use different parameter names:
        - differential_evolution: maxiter
        - dual_annealing: maxiter
        - basinhopping: niter
        - shgo: iters
        - direct: maxfun

        Returns
        -------
        str
            The parameter name for iteration control.
            Default is "maxiter".
        """
        return "maxiter"

    def _convert_to_scipy_space(self, param_space):
        """Convert Hyperactive parameter space to scipy bounds format.

        Validates that all parameters are continuous (tuples) and converts
        to scipy's bounds format: list of (low, high) tuples.

        Parameters
        ----------
        param_space : dict[str, tuple]
            The parameter space to convert. Keys are parameter names,
            values must be tuples of (low, high) for continuous ranges.

        Returns
        -------
        bounds : list of tuple
            Scipy-compatible bounds as [(low, high), ...].
        param_names : list of str
            Parameter names in the order matching bounds.

        Raises
        ------
        ValueError
            If parameter space contains non-tuple values (lists, arrays).

        Examples
        --------
        >>> adapter = _BaseScipyAdapter()
        >>> space = {"x": (0.0, 1.0), "y": (-5.0, 5.0)}
        >>> bounds, names = adapter._convert_to_scipy_space(space)
        >>> bounds
        [(0.0, 1.0), (-5.0, 5.0)]
        >>> names
        ['x', 'y']
        """
        bounds = []
        param_names = []

        for key, space in param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                low, high = space
                bounds.append((float(low), float(high)))
                param_names.append(key)
            elif isinstance(space, (list, np.ndarray)):
                raise ValueError(
                    f"Scipy optimizers only support continuous parameter spaces. "
                    f"Parameter '{key}' has discrete values (list/array). "
                    f"Use optuna or gfo backends for discrete/categorical parameters."
                )
            else:
                raise ValueError(
                    f"Unsupported parameter space type for '{key}': {type(space)}. "
                    f"Expected tuple (low, high) for continuous range."
                )

        return bounds, param_names

    def _array_to_dict(self, x_array, param_names):
        """Convert scipy array to Hyperactive parameter dictionary.

        Parameters
        ----------
        x_array : np.ndarray
            Array of parameter values from scipy optimizer.
        param_names : list of str
            Parameter names in order matching x_array.

        Returns
        -------
        dict
            Parameter dictionary with names as keys.
        """
        return dict(zip(param_names, x_array))

    def _get_x0_from_initialize(self, bounds, param_names):
        """Extract initial point from initialize configuration.

        Parameters
        ----------
        bounds : list of tuple
            Scipy bounds as [(low, high), ...].
        param_names : list of str
            Parameter names in order.

        Returns
        -------
        np.ndarray or None
            Initial point if warm_start provided, else None.
        """
        if self.initialize is None:
            return None

        if not isinstance(self.initialize, dict):
            return None

        warm_start = self.initialize.get("warm_start")
        if warm_start is None or not isinstance(warm_start, list):
            return None

        if len(warm_start) == 0:
            return None

        # Use first warm start point
        point = warm_start[0]
        x0 = np.array([point.get(name, (b[0] + b[1]) / 2)
                       for name, b in zip(param_names, bounds)])
        return x0

    def _create_callback(self, start_time, max_time):
        """Create a callback for time-based early stopping.

        Parameters
        ----------
        start_time : float
            Start time from time.time().
        max_time : float or None
            Maximum time in seconds, or None for no limit.

        Returns
        -------
        callable or None
            Callback function that returns True to stop, or None.
        """
        if max_time is None:
            return None

        def callback(*args, **kwargs):
            elapsed = time.time() - start_time
            return elapsed > max_time

        return callback

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the scipy optimization.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Number of iterations.
        max_time : float, optional
            Maximum time in seconds.
        **kwargs
            Additional parameters (unused, for compatibility).

        Returns
        -------
        dict
            Best parameters found during optimization.
        """
        # Convert search space
        bounds, param_names = self._convert_to_scipy_space(param_space)

        # Create objective function (negated for minimization)
        def objective(x):
            params = self._array_to_dict(x, param_names)
            score = experiment(params)
            return -score  # Negate for scipy minimization

        # Get scipy function and kwargs
        scipy_func = self._get_scipy_func()
        opt_kwargs = self._get_optimizer_kwargs()

        # Set iteration parameter
        iter_param = self._get_iteration_param_name()
        opt_kwargs[iter_param] = n_iter

        # Set random state if provided
        if self.random_state is not None and "seed" not in opt_kwargs:
            opt_kwargs["seed"] = self.random_state

        # Set up callback for time limit
        start_time = time.time()
        callback = self._create_callback(start_time, max_time)
        if callback is not None:
            opt_kwargs["callback"] = callback

        # Get initial point from warm start if available
        x0 = self._get_x0_from_initialize(bounds, param_names)
        if x0 is not None and "x0" not in opt_kwargs:
            opt_kwargs["x0"] = x0

        # Run optimization
        result = scipy_func(objective, bounds, **opt_kwargs)

        # Extract best parameters
        best_params = self._array_to_dict(result.x, param_names)
        self.best_score_ = -result.fun  # Negate back to maximization

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

        ackley_exp = Ackley.create_test_instance()

        # Test with continuous ranges
        params_continuous = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
        }

        return [params_continuous]
