"""Adapter for gfo package."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from skbase.utils.stdout_mute import StdoutMute
from collections.abc import Mapping, Sequence
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

        search_config["search_space"] = self._to_dict_np(search_config["search_space"])

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
        This method coerces lists or tuples in the search space to numpy arrays.

        Parameters
        ----------
        search_space : dict with str keys and iterable values
            The search space to coerce.

        Returns
        -------
        dict with str keys and 1D numpy arrays as values
            The coerced search space.
        """
        import numpy as np

        normalized = self._normalize_search_space(search_space)

        def coerce_to_numpy(values, param_name):
            """Coerce a list or tuple to a numpy array."""
            arr = np.asarray(values)
            if arr.ndim != 1:
                raise ValueError(
                    "Search space values for parameter "
                    f"'{param_name}' must be 1-dimensional array-like; got "
                    f"shape {arr.shape}."
                )
            return arr
        
        def convert(grid):
            return {key: coerce_to_numpy(val, key) for key, val in grid.items()}

        if isinstance(normalized, list):
            return [convert(grid) for grid in normalized]

        return convert(normalized)

    def _normalize_search_space(self, search_space):

        if search_space is None:
            return None

        if isinstance(search_space, Mapping):
            return dict(search_space)

        from sklearn.model_selection import ParameterGrid as parameter_grid

        if isinstance(search_space, parameter_grid):

            grids = [dict(grid) for grid in search_space.param_grid]

        elif isinstance(search_space, Sequence) and not isinstance(search_space, (str,bytes)):

            grids = [dict(grid) for grid in search_space]
        
        else:

            raise TypeError(
                f"search space must be dict, list/tuple of dict, or sklearn.model_selection.ParameterGrid"
                f"({type(search_space).__name__})."
            )

        if len(grids) == 1:
            return grids[0]
        else:
            return grids


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

        search_space = search_config.pop("search_space", None)

        if isinstance(search_space, list):
            search_spaces = search_space
        else:
            search_spaces = [search_space]

        gfo_cls = self._get_gfo_class()

        best_score = None
        best_params = None

        for grid in search_spaces:
            grid_config = dict(search_config)
            grid_config["search_space"] = grid

            gfopt = gfo_cls(**grid_config)

            with StdoutMute(active=not self.verbose):
                gfopt.search(
                    objective_function=experiment.score,
                    n_iter=n_iter,
                    max_time=max_time,
                )

            grid_params = gfopt.best_para
            grid_score = getattr(gfopt, "best_score", None)
            if grid_score is None and grid_params is not None:
                grid_score = experiment.score(grid_params)[0]

            if best_params is None or (grid_score is not None and grid_score > best_score):
                best_params = grid_params
                best_score = grid_score
                
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
