"""Adapter for gfo package."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from skbase.utils.stdout_mute import StdoutMute

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseGFOadapter"]


class _BaseGFOadapter(BaseOptimizer):
    """Adapter base class for gradient-free-optimizers.

    * default tag setting
    * default _run method
    * default get_search_config
    * default get_test_params
    * Handles defaults for "initialize" parameter
    * extension interface: _get_gfo_class, docstring, tags
    """

    _tags = {
        "authors": "SimonBlanke",
        "python_dependencies": ["gradient-free-optimizers>=1.5.0"],
        # search space capabilities
        "capability:search_space:continuous": True,  # via discretization
        "capability:search_space:discrete": True,
        "capability:search_space:categorical": True,
        "capability:search_space:mixed": True,
        "capability:search_space:log_scale": True,  # via log-discretization
        "capability:search_space:conditional": False,  # handled at Hyperactive level
        "capability:search_space:constraints": True,  # native GFO support
        "capability:search_space:distributions": True,  # via sampling
        "capability:search_space:nested": False,  # requires conditional
    }

    def __init__(self):
        super().__init__()

        if self.initialize is None:
            self._initialize = {"grid": 4, "random": 2, "vertices": 4}
        else:
            self._initialize = self.initialize

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def get_search_config(self):
        """Get the search configuration.

        Returns
        -------
        dict with str keys
            The search configuration dictionary.
        """
        search_config = super().get_search_config()
        search_config["initialize"] = self._initialize
        del search_config["verbose"]

        search_config = self._handle_gfo_defaults(search_config)

        # Extract constraints from SearchSpace before converting
        original_space = search_config["search_space"]
        space_constraints = self._get_constraints_from_search_space(original_space)

        # Merge constraints: SearchSpace constraints + explicit constraints
        if space_constraints:
            existing = search_config.get("constraints") or []
            search_config["constraints"] = existing + space_constraints

        search_config["search_space"] = self._to_dict_np(original_space)

        return search_config

    def _handle_gfo_defaults(self, search_config):
        """Handle default values for GFO search configuration.

        Temporary measure until GFO handles defaults gracefully.

        Parameters
        ----------
        search_config : dict with str keys
            The search configuration dictionary to handle defaults for.

        Returns
        -------
        search_config : dict with str keys
            The search configuration dictionary with defaults handled.
        """
        if "sampling" in search_config and search_config["sampling"] is None:
            search_config["sampling"] = {"random": 1000000}

        if "tree_para" in search_config and search_config["tree_para"] is None:
            search_config["tree_para"] = {"n_estimators": 100}

        return search_config

    def _to_dict_np(self, search_space):
        """Coerce the search space to a format suitable for gfo optimizers.

        gfo expects dicts of numpy arrays, not lists.
        This method handles both dict-based search spaces and SearchSpace objects.

        Parameters
        ----------
        search_space : dict or SearchSpace
            The search space to coerce. Can be either a dict with str keys
            and iterable values, or a SearchSpace object.

        Returns
        -------
        dict with str keys and 1D numpy arrays as values
            The coerced search space.
        """
        import numpy as np

        # Check if it's a SearchSpace object
        from hyperactive.search_space import SearchSpace

        if isinstance(search_space, SearchSpace):
            # Use the GFO adapter to convert
            return search_space.to_backend("gfo")

        # Original dict-based handling
        def coerce_to_numpy(arr):
            """Coerce a list or tuple to a numpy array."""
            if not isinstance(arr, np.ndarray):
                return np.array(arr)
            return arr

        coerced_search_space = {k: coerce_to_numpy(v) for k, v in search_space.items()}
        return coerced_search_space

    def _get_constraints_from_search_space(self, search_space):
        """Extract constraints from SearchSpace if available.

        Parameters
        ----------
        search_space : dict or SearchSpace
            The search space.

        Returns
        -------
        list or None
            List of constraint functions (may be empty), or None if
            search_space is not a SearchSpace object.
        """
        from hyperactive.search_space import SearchSpace

        if isinstance(search_space, SearchSpace):
            from hyperactive.search_space.adapters import GFOSearchSpaceAdapter

            adapter = GFOSearchSpaceAdapter(search_space)
            return adapter.get_constraints()
        return None

    def _solve(self, experiment, **search_config):
        """Run the optimization search process.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            identical to return of ``get_search_config``.

        Returns
        -------
        dict with str keys
            The best parameters found during the search.
            Must have keys a subset or identical to experiment.paramnames().
        """
        n_iter = search_config.pop("n_iter", 100)
        max_time = search_config.pop("max_time", None)

        gfo_cls = self._get_gfo_class()
        gfopt = gfo_cls(**search_config)

        with StdoutMute(active=not self.verbose):
            gfopt.search(
                objective_function=experiment.score,
                n_iter=n_iter,
                max_time=max_time,
            )
        best_params = gfopt.best_para
        return best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import numpy as np

        from hyperactive.experiment.integrations import SklearnCvExperiment

        sklearn_exp = SklearnCvExperiment.create_test_instance()
        params_sklearn = {
            "experiment": sklearn_exp,
            "search_space": {
                "C": np.array([0.01, 0.1, 1, 10]),
                "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
            },
            "n_iter": 100,
        }

        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()
        params_ackley = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": np.linspace(-5, 5, 10),
                "x1": np.linspace(-5, 5, 10),
            },
            "n_iter": 100,
        }
        params_ackley_list = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": list(np.linspace(-5, 5, 10)),
                "x1": list(np.linspace(-5, 5, 10)),
            },
            "n_iter": 100,
        }
        return [params_sklearn, params_ackley, params_ackley_list]
