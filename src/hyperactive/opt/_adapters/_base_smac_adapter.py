"""Base adapter for SMAC3 optimizers."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import time

import numpy as np

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseSMACAdapter"]


class _BaseSMACAdapter(BaseOptimizer):
    """Base adapter class for SMAC3 optimizers.

    This adapter handles the conversion between Hyperactive's interface and
    SMAC3's facade-based optimization pattern. Key responsibilities:

    * Search space conversion to ConfigSpace format
    * Score negation (SMAC minimizes, Hyperactive maximizes)
    * Ask-tell optimization loop implementation
    * Random state handling
    * Time-based early stopping

    Extension interface for subclasses:

    * ``_get_facade_class``: Return the SMAC facade class
    * ``_get_facade_kwargs``: Return facade-specific constructor kwargs
    * ``_get_scenario_kwargs``: Return scenario-specific kwargs

    Notes
    -----
    SMAC3 is designed for minimization, while Hyperactive uses maximization
    (higher scores are better). This adapter negates scores when calling
    ``smac.tell()`` to handle this difference.

    SMAC3 uses ConfigSpace for parameter space definition. This adapter
    converts Hyperactive's simple dict format to ConfigSpace objects.

    Parameter type detection uses the following rules:

    * Tuple ``(int, int)``: Integer parameter
    * Tuple ``(float, float)`` or mixed: Float parameter
    * List/array: Categorical parameter

    For ambiguous cases like ``(1, 10)``, the adapter checks if both bounds
    are Python ``int`` type. Use ``(1.0, 10.0)`` to force float interpretation.
    """

    _tags = {
        "python_dependencies": ["smac", "ConfigSpace"],
        "info:name": "SMAC3-based optimizer",
    }

    def __init__(
        self,
        param_space=None,
        n_iter=100,
        max_time=None,
        initialize=None,
        random_state=None,
        deterministic=True,
        experiment=None,
    ):
        self.param_space = param_space
        self.n_iter = n_iter
        self.max_time = max_time
        self.initialize = initialize
        self.random_state = random_state
        self.deterministic = deterministic
        self.experiment = experiment
        super().__init__()

    def _get_facade_class(self):
        """Get the SMAC facade class to use.

        Returns
        -------
        class
            The SMAC facade class. Must be a class from ``smac``.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_facade_class to return "
            "the SMAC facade class."
        )

    def _get_facade_kwargs(self):
        """Get facade-specific keyword arguments for instantiation.

        Override this method in subclasses to pass algorithm-specific
        parameters to the SMAC facade constructor.

        Returns
        -------
        dict
            Keyword arguments to pass to the facade constructor.
            Default is an empty dict.
        """
        return {}

    def _get_scenario_kwargs(self):
        """Get scenario-specific keyword arguments.

        Override this method in subclasses to pass scenario-specific
        parameters (e.g., min_budget, max_budget for multi-fidelity).

        Returns
        -------
        dict
            Keyword arguments to pass to the Scenario constructor.
            Default is an empty dict.
        """
        return {}

    def _convert_to_configspace(self, param_space):
        """Convert Hyperactive parameter space to ConfigSpace.

        Handles conversion of different parameter formats:

        * Tuples ``(int, int)``: Converted to ``Integer``
        * Tuples ``(float, float)`` or mixed: Converted to ``Float``
        * Lists/arrays of values: Converted to ``Categorical``
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
        ConfigurationSpace
            ConfigSpace ConfigurationSpace object.

        Raises
        ------
        ValueError
            If parameter space format is not supported.

        Examples
        --------
        >>> adapter = _BaseSMACAdapter()
        >>> space = {"x": (0.0, 1.0), "y": [1, 2, 3], "z": (1, 10)}
        >>> cs = adapter._convert_to_configspace(space)
        """
        from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

        cs = ConfigurationSpace(seed=self.random_state)

        for name, space in param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                low, high = space
                # Check if both bounds are strictly int type
                if isinstance(low, int) and isinstance(high, int):
                    # Exclude bool since bool is subclass of int
                    if not isinstance(low, bool) and not isinstance(high, bool):
                        cs.add(Integer(name, bounds=(low, high)))
                    else:
                        # bool values -> treat as categorical
                        cs.add(Categorical(name, items=[low, high]))
                else:
                    # Float range (includes mixed int/float like (1, 10.0))
                    cs.add(Float(name, bounds=(float(low), float(high))))

            elif isinstance(space, (list, np.ndarray)):
                values = list(space) if isinstance(space, np.ndarray) else space

                if len(values) == 0:
                    raise ValueError(f"Empty parameter space for '{name}'")

                cs.add(Categorical(name, items=values))

            else:
                raise ValueError(
                    f"Unsupported parameter space type for '{name}': {type(space)}. "
                    "Expected tuple (low, high) or list of values."
                )

        return cs

    def _config_to_dict(self, config):
        """Convert SMAC Configuration to parameter dictionary.

        Converts numpy scalar types to native Python types to ensure
        compatibility with sklearn estimators and JSON serialization.

        Parameters
        ----------
        config : Configuration
            SMAC Configuration object.

        Returns
        -------
        dict
            Parameter dictionary with keys matching param_space.
            All values are native Python types (str, int, float, bool).
        """
        params = dict(config)
        # Convert numpy scalars to Python native types
        for key, value in params.items():
            if hasattr(value, "item"):
                # numpy scalar types have .item() method
                params[key] = value.item()
        return params

    def _create_target_function(self, experiment):
        """Create a target function for SMAC optimization.

        SMAC expects a target function with signature:
        ``target_function(config, seed=None) -> float``

        This method creates such a function that wraps the experiment.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.

        Returns
        -------
        callable
            Target function compatible with SMAC.
        """

        def target_function(config, seed=None):
            params = self._config_to_dict(config)
            score = experiment(params)
            # Negate score since SMAC minimizes and Hyperactive maximizes
            return -score

        return target_function

    def _setup_warm_start(self, smac, experiment, initialize):
        """Set up warm start initialization if provided.

        Warm start points are evaluated and told to SMAC before
        the main optimization loop begins.

        Parameters
        ----------
        smac : AbstractFacade
            The SMAC facade instance.
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
        from smac.runhistory.dataclasses import TrialValue

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

            # Ask for a trial info and tell SMAC about this evaluation
            info = smac.ask()

            # Tell optimizer about this evaluation (negate for minimization)
            value = TrialValue(cost=-score, time=0.0)
            smac.tell(info, value)
            count += 1

        return count

    def _solve(self, experiment, param_space, n_iter, max_time=None, **kwargs):
        """Run the SMAC optimization loop.

        Implements the ask-tell pattern:

        1. Create ConfigSpace from param_space
        2. Create Scenario with budget and constraints
        3. Create facade with target function
        4. Run optimization loop
        5. Return best parameters

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        param_space : dict
            The parameter space to search.
        n_iter : int
            Number of iterations (trials).
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
        from smac import Scenario
        from smac.runhistory.dataclasses import TrialValue

        # Convert search space to ConfigSpace format
        configspace = self._convert_to_configspace(param_space)

        # Build scenario kwargs
        scenario_kwargs = {
            "configspace": configspace,
            "n_trials": n_iter,
            "deterministic": self.deterministic,
        }

        # Add seed if provided
        if self.random_state is not None:
            scenario_kwargs["seed"] = self.random_state

        # Add time limit if provided
        if max_time is not None:
            scenario_kwargs["walltime_limit"] = max_time

        # Add subclass-specific scenario kwargs
        scenario_kwargs.update(self._get_scenario_kwargs())

        # Create scenario
        scenario = Scenario(**scenario_kwargs)

        # Get facade class and kwargs
        facade_cls = self._get_facade_class()
        facade_kwargs = self._get_facade_kwargs()

        # Create target function
        target_function = self._create_target_function(experiment)

        # Create facade instance
        smac = facade_cls(
            scenario=scenario,
            target_function=target_function,
            overwrite=True,  # Allow overwriting previous runs
            **facade_kwargs,
        )

        # Handle warm start initialization
        warm_start_count = self._setup_warm_start(smac, experiment, self.initialize)

        # Track best result manually for early access
        best_score = float("-inf")
        best_params = None

        # Optimization loop using ask-tell interface
        start_time = time.time()
        remaining_budget = n_iter - warm_start_count

        for _ in range(remaining_budget):
            # Check time limit
            if max_time is not None and (time.time() - start_time) > max_time:
                break

            # Ask for next configuration
            info = smac.ask()

            # Extract parameters and evaluate
            params = self._config_to_dict(info.config)
            score = experiment(params)

            # Tell SMAC (negate for minimization)
            value = TrialValue(cost=-score, time=0.0)
            smac.tell(info, value)

            # Track best
            if score > best_score:
                best_score = score
                best_params = params.copy()

        # Get incumbent (best found configuration) from intensifier
        incumbent = smac.intensifier.get_incumbent()
        if incumbent is not None:
            final_params = self._config_to_dict(incumbent)
        else:
            # Fallback to manually tracked best
            final_params = best_params

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

        Notes
        -----
        Test parameter sets cover:

        * Float tuple ranges (continuous parameters)
        * Integer tuple ranges (discrete parameters)
        * Categorical lists (string and numeric)
        * Boolean categorical parameters
        * Mixed parameter types (float + int + categorical)
        * Warm start initialization
        * Random state for reproducibility
        * Deterministic vs non-deterministic settings
        """
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        X, y = load_iris(return_X_y=True)
        sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
        ackley_exp = Ackley.create_test_instance()

        # Test 1: Float tuple ranges (continuous parameters)
        params_float_tuples = {
            "param_space": {
                "C": (0.01, 10.0),
                "gamma": (0.0001, 1.0),
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test 2: Categorical lists (discrete values)
        params_categorical = {
            "param_space": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "gamma": [0.0001, 0.001, 0.01, 0.1],
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test 3: Integer tuple ranges
        params_integer_tuples = {
            "param_space": {
                "x0": (-5, 5),
                "x1": (-5, 5),
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }

        # Test 4: Mixed parameter types (float + categorical)
        params_mixed_float_cat = {
            "param_space": {
                "C": (0.01, 100.0),
                "gamma": (0.0001, 1.0),
                "kernel": ["rbf", "linear", "poly"],
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test 5: Mixed parameter types (int + float + categorical)
        params_mixed_all = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": [1, 2, 3, 4, 5],
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }

        # Test 6: Boolean categorical parameters
        params_boolean_cat = {
            "param_space": {
                "C": (0.1, 10.0),
                "shrinking": [True, False],
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test 7: String categorical with many options
        params_string_cat = {
            "param_space": {
                "C": (0.1, 10.0),
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
            },
            "n_iter": 10,
            "experiment": sklearn_exp,
        }

        # Test 8: With random_state for reproducibility
        params_random_state = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "random_state": 42,
            "experiment": ackley_exp,
        }

        # Test 9: With deterministic=False (stochastic objective)
        params_non_deterministic = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 10,
            "deterministic": False,
            "experiment": ackley_exp,
        }

        # Test 10: With warm_start initialization
        params_warm_start = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "initialize": {"warm_start": [{"x0": 0.0, "x1": 0.0}]},
            "experiment": ackley_exp,
        }

        # Test 11: Multiple warm start points
        params_multi_warm_start = {
            "param_space": {
                "x0": (-5.0, 5.0),
                "x1": (-5.0, 5.0),
            },
            "n_iter": 15,
            "initialize": {
                "warm_start": [
                    {"x0": 0.0, "x1": 0.0},
                    {"x0": 1.0, "x1": -1.0},
                    {"x0": -2.0, "x1": 2.0},
                ]
            },
            "random_state": 123,
            "experiment": ackley_exp,
        }

        # Test 12: Large integer range
        params_large_int_range = {
            "param_space": {
                "x0": (-100, 100),
                "x1": (0, 1000),
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }

        # Test 13: Small float range (high precision)
        params_small_float_range = {
            "param_space": {
                "x0": (-0.001, 0.001),
                "x1": (-0.001, 0.001),
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }

        # Test 14: Asymmetric ranges
        params_asymmetric = {
            "param_space": {
                "x0": (-10.0, 2.0),
                "x1": (0.5, 100.0),
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }

        return [
            params_float_tuples,
            params_categorical,
            params_integer_tuples,
            params_mixed_float_cat,
            params_mixed_all,
            params_boolean_cat,
            params_string_cat,
            params_random_state,
            params_non_deterministic,
            params_warm_start,
            params_multi_warm_start,
            params_large_int_range,
            params_small_float_range,
            params_asymmetric,
        ]
