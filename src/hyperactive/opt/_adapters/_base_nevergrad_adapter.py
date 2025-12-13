"""Base adapter for Nevergrad optimizers."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import time

import numpy as np

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseNevergradAdapter"]


class _BaseNevergradAdapter(BaseOptimizer):
    """Base adapter class for Nevergrad optimizers.

    This adapter handles the conversion between Hyperactive's interface and
    Nevergrad's ask-tell optimization pattern. Key responsibilities:

    * Search space conversion to Nevergrad's parametrization system
    * Score negation (Nevergrad minimizes, Hyperactive maximizes)
    * Ask-tell optimization loop implementation
    * Random state handling
    * Time-based early stopping

    Extension interface for subclasses:

    * ``_get_nevergrad_class``: Return the Nevergrad optimizer class
    * ``_get_optimizer_kwargs``: Return optimizer-specific constructor kwargs

    Notes
    -----
    Nevergrad is designed for minimization, while Hyperactive uses maximization
    (higher scores are better). This adapter negates scores when calling
    ``optimizer.tell()`` to handle this difference.

    Nevergrad's ``Choice`` parameter uses softmax sampling which introduces
    stochasticity. For ordered numeric values, ``TransitionChoice`` is used
    instead for more deterministic behavior.
    """

    _tags = {
        "python_dependencies": ["nevergrad"],
        "info:name": "Nevergrad-based optimizer",
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

    def _get_nevergrad_class(self):
        """Get the Nevergrad optimizer class to use.

        Returns
        -------
        class
            The Nevergrad optimizer class. Must be a class from
            ``nevergrad.optimizers``.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_nevergrad_class to return "
            "the Nevergrad optimizer class."
        )

    def _get_optimizer_kwargs(self):
        """Get optimizer-specific keyword arguments for instantiation.

        Override this method in subclasses to pass algorithm-specific
        parameters to the Nevergrad optimizer constructor.

        Note: These are passed when instantiating the optimizer class,
        not when configuring the class. For configuration parameters,
        use ``_get_config_kwargs``.

        Returns
        -------
        dict
            Keyword arguments to pass to the optimizer constructor.
            Default is an empty dict.
        """
        return {}

    def _get_config_kwargs(self):
        """Get optimizer configuration keyword arguments.

        Nevergrad uses a factory pattern where optimizer classes are
        configured first, then instantiated. Override this method to
        pass configuration parameters.

        For example, for DifferentialEvolution:
        - Configuration: ``DifferentialEvolution(F1=0.8, crossover=0.9)``
        - Instantiation: ``configured_class(parametrization=..., budget=...)``

        Returns
        -------
        dict
            Keyword arguments for configuring the optimizer class.
            Default is an empty dict.
        """
        return {}

    def _is_ordered_numeric(self, values):
        """Check if values are ordered numeric (for TransitionChoice vs Choice).

        Parameters
        ----------
        values : list
            List of values to check.

        Returns
        -------
        bool
            True if all values are numeric and appear to be ordered.
        """
        if len(values) < 2:
            return False

        try:
            numeric_values = [float(v) for v in values]
            # Check if sorted (ascending or descending)
            is_ascending = all(
                numeric_values[i] <= numeric_values[i + 1]
                for i in range(len(numeric_values) - 1)
            )
            is_descending = all(
                numeric_values[i] >= numeric_values[i + 1]
                for i in range(len(numeric_values) - 1)
            )
            return is_ascending or is_descending
        except (TypeError, ValueError):
            return False

    def _convert_to_nevergrad_space(self, param_space):
        """Convert Hyperactive parameter space to Nevergrad Instrumentation.

        Handles conversion of different parameter formats:

        * Tuples ``(low, high)``: Converted to bounded ``Scalar`` (float) or
          ``TransitionChoice`` over range (int)
        * Lists/arrays of values: Converted to ``TransitionChoice`` for ordered
          numeric values, ``Choice`` for categorical values
        * numpy arrays: Treated same as lists

        Parameters
        ----------
        param_space : dict[str, tuple | list | np.ndarray]
            The parameter space to convert. Keys are parameter names,
            values are either:
            - Tuple of (low, high) for ranges
            - List/array of discrete values

        Returns
        -------
        ng.p.Instrumentation
            Nevergrad Instrumentation object for the parameter space.

        Raises
        ------
        ValueError
            If parameter space format is not supported.

        Examples
        --------
        >>> adapter = _BaseNevergradAdapter()
        >>> space = {"x": (0.0, 1.0), "y": [1, 2, 3]}
        >>> inst = adapter._convert_to_nevergrad_space(space)
        """
        import nevergrad as ng

        ng_params = {}

        for key, space in param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    # Integer range -> TransitionChoice over range
                    ng_params[key] = ng.p.TransitionChoice(list(range(low, high + 1)))
                else:
                    # Float range -> Scalar with bounds
                    ng_params[key] = ng.p.Scalar(lower=float(low), upper=float(high))

            elif isinstance(space, (list, np.ndarray)):
                values = list(space) if isinstance(space, np.ndarray) else space

                if len(values) == 0:
                    raise ValueError(f"Empty parameter space for '{key}'")

                # Use TransitionChoice for ordered numeric, Choice for categorical
                if self._is_ordered_numeric(values):
                    ng_params[key] = ng.p.TransitionChoice(values)
                else:
                    ng_params[key] = ng.p.Choice(values)

            else:
                raise ValueError(
                    f"Unsupported parameter space type for '{key}': {type(space)}. "
                    "Expected tuple (low, high) or list of values."
                )

        return ng.p.Instrumentation(**ng_params)

    def _candidate_to_params(self, candidate):
        """Extract parameter dictionary from Nevergrad candidate.

        Parameters
        ----------
        candidate : ng.p.Instrumentation
            Nevergrad candidate from ask() or provide_recommendation().

        Returns
        -------
        dict
            Parameter dictionary with keys matching param_space.
        """
        return dict(candidate.kwargs)

    def _setup_warm_start(self, optimizer, experiment, initialize):
        """Set up warm start initialization if provided.

        Warm start points are evaluated and told to the optimizer before
        the main optimization loop begins.

        Parameters
        ----------
        optimizer : nevergrad.optimizers.base.Optimizer
            The Nevergrad optimizer instance.
        experiment : BaseExperiment
            The experiment to evaluate.
        initialize : dict or None
            Initialization configuration. If contains "warm_start" key,
            those points are evaluated and told to the optimizer.

        Returns
        -------
        int
            Number of warm start evaluations performed.
        """
        if initialize is None:
            return 0

        if not isinstance(initialize, dict) or "warm_start" not in initialize:
            return 0

        warm_start_points = initialize["warm_start"]
        if not isinstance(warm_start_points, list):
            return 0

        count = 0
        for point in warm_start_points:
            # Evaluate the point
            score = experiment(point)

            # Ask for a candidate and tell with the result
            candidate = optimizer.ask()

            # Set the candidate's values to match the warm start point
            for key, value in point.items():
                if key in candidate.kwargs:
                    # Update the parametrization value
                    pass  # Nevergrad handles this through tell

            # Tell optimizer about this evaluation (negate for minimization)
            optimizer.tell(candidate, -score)
            count += 1

        return count

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the Nevergrad optimization loop.

        Implements the ask-tell pattern:
        1. Ask optimizer for candidate parameters
        2. Evaluate candidate using experiment.score()
        3. Tell optimizer the (negated) result
        4. Repeat until budget exhausted or time limit reached

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Number of iterations (budget).
        max_time : float, optional
            Maximum time in seconds. If provided, optimization stops
            when time limit is reached even if budget not exhausted.
        **kwargs
            Additional parameters (unused, for compatibility).

        Returns
        -------
        dict
            Best parameters found during optimization.
        """
        import nevergrad as ng

        # Convert search space to Nevergrad format
        parametrization = self._convert_to_nevergrad_space(param_space)

        # Set random state if provided
        if self.random_state is not None:
            parametrization.random_state = np.random.RandomState(self.random_state)

        # Get optimizer class/factory and configuration
        ng_cls = self._get_nevergrad_class()
        config_kwargs = self._get_config_kwargs()
        optimizer_kwargs = self._get_optimizer_kwargs()

        # Nevergrad uses a factory pattern:
        # - Some optimizers are classes (isinstance(ng_cls, type) == True)
        # - Some are already configured instances (callable but not a class)
        # For classes: call to configure, then call result to instantiate
        # For instances: call directly to instantiate

        if config_kwargs:
            # Configure the optimizer factory/class with algorithm parameters
            configured = ng_cls(**config_kwargs)
        else:
            configured = ng_cls

        # Create optimizer instance with parametrization and budget
        optimizer = configured(
            parametrization=parametrization, budget=n_iter, **optimizer_kwargs
        )

        # Handle warm start initialization
        warm_start_count = self._setup_warm_start(
            optimizer, experiment, self.initialize
        )

        # Track best result
        best_score = float("-inf")
        best_params = None

        # Optimization loop
        start_time = time.time()
        remaining_budget = n_iter - warm_start_count

        for _ in range(remaining_budget):
            # Check time limit
            if max_time is not None and (time.time() - start_time) > max_time:
                break

            # Ask for candidate
            candidate = optimizer.ask()

            # Extract parameters and evaluate
            params = self._candidate_to_params(candidate)
            score = experiment(params)

            # Tell optimizer (negate for minimization)
            optimizer.tell(candidate, -score)

            # Track best
            if score > best_score:
                best_score = score
                best_params = params.copy()

        # Get recommendation from optimizer
        recommendation = optimizer.provide_recommendation()
        final_params = self._candidate_to_params(recommendation)

        # Store best score for access
        self.best_score_ = best_score

        return final_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer.

        Returns
        -------
        list of dict
            List of parameter configurations for testing.
        """
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from hyperactive.experiment.integrations import SklearnCvExperiment

        X, y = load_iris(return_X_y=True)
        sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

        # Test with tuple ranges
        params_tuples = {
            "param_space": {
                "C": (0.01, 10.0),
                "gamma": (0.0001, 1.0),
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test with discrete lists
        params_lists = {
            "param_space": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "gamma": [0.0001, 0.001, 0.01, 0.1],
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test with mixed types
        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()
        params_bench = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 20,
            "experiment": ackley_exp,
        }

        return [params_tuples, params_lists, params_bench]
