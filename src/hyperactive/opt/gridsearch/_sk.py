"""Grid search optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import ParameterGrid

from hyperactive.base import BaseOptimizer
from hyperactive.opt._common import _score_params
from hyperactive.utils.parallel import parallelize


class GridSearchSk(BaseOptimizer):
    """Grid search optimizer, with backend selection and sklearn style parameter grid.

    Parameters
    ----------
    param_grid : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.

    error_score : float, default=np.nan
        The score to assign if an error occurs during the evaluation of a parameter set.

    backend : {"dask", "loky", "multiprocessing", "threading", "ray"}, default = "None".
        Parallelization backend to use in the search process.

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Example
    -------
    Grid search applied to scikit-learn parameter tuning:

    1. defining the experiment to optimize:
    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>>
    >>> sklearn_exp = SklearnCvExperiment(
    ...     estimator=SVC(),
    ...     X=X,
    ...     y=y,
    ... )

    2. setting up the grid search optimizer:
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>> param_grid = {
    ...     "C": [0.01, 0.1, 1, 10],
    ...     "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ... }
    >>> grid_search = GridSearch(param_grid, experiment=sklearn_exp)

    3. running the grid search:
    >>> best_params = grid_search.solve()

    Best parameters can also be accessed via the attributes:
    >>> best_params = grid_search.best_params_

    To parallelize the search, set the ``backend`` and ``backend_params``:
    >>> grid_search = GridSearch(
    ...     param_grid,
    ...     backend="joblib",
    ...     backend_params={"n_jobs": -1},
    ...     experiment=sklearn_exp,
    ... )
    """

    def __init__(
        self,
        unified_space=None,
        param_grid=None,
        error_score=np.nan,
        backend="None",
        backend_params=None,
        experiment=None,
    ):
        self.experiment = experiment
        self.unified_space = unified_space
        self.param_grid = param_grid
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

    def get_search_config(self):
        """Get the search configuration.

        Returns
        -------
        dict with str keys
            The search configuration dictionary.
        """
        search_config = super().get_search_config()

        # Resolve: unified_space is converted to param_grid
        unified_space = search_config.pop("unified_space", None)
        param_grid = search_config.get("param_grid")

        # Validate: only one should be set
        if unified_space is not None and param_grid is not None:
            raise ValueError(
                "Provide either 'unified_space' or 'param_grid', not both. "
                "Use 'unified_space' for simple dict[str, list] format, "
                "or 'param_grid' for native sklearn format."
            )

        # Use unified_space if param_grid is not set
        if unified_space is not None:
            search_config["param_grid"] = unified_space

        return search_config

    def _check_param_grid(self, param_grid):
        """_check_param_grid from sklearn 1.0.2, before it was removed."""
        if hasattr(param_grid, "items"):
            param_grid = [param_grid]

        for p in param_grid:
            for name, v in p.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                    raise ValueError(
                        f"Parameter grid for parameter ({name}) needs to"
                        f" be a list or numpy array, but got ({type(v)})."
                        " Single values need to be wrapped in a list"
                        " with one element."
                    )

                if len(v) == 0:
                    raise ValueError(
                        f"Parameter values for parameter ({name}) need "
                        "to be a non-empty sequence."
                    )

    def _solve(self, experiment, param_grid, error_score, backend, backend_params):
        """Run the optimization search process."""
        self._check_param_grid(param_grid)
        candidate_params = list(ParameterGrid(param_grid))

        meta = {
            "experiment": experiment,
            "error_score": error_score,
        }

        # scores are sign-adjusted via experiment.score (higher-is-better)
        scores = parallelize(
            fun=_score_params,
            iter=candidate_params,
            meta=meta,
            backend=backend,
            backend_params=backend_params,
        )

        # select best by maximizing standardized score
        best_index = int(np.argmax(scores))

        best_params = candidate_params[best_index]

        # store public attributes
        self.best_index_ = best_index
        self.best_score_ = float(scores[best_index])
        self.best_params_ = best_params

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
        from hyperactive.experiment.integrations import SklearnCvExperiment

        sklearn_exp = SklearnCvExperiment.create_test_instance()
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.0001, 0.01, 0.1, 1, 10],
        }
        params_sklearn = {
            "experiment": sklearn_exp,
            "param_grid": param_grid,
        }

        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()
        param_grid = {
            "x0": np.linspace(-5, 5, 10),
            "x1": np.linspace(-5, 5, 10),
        }
        params_ackley = {
            "experiment": ackley_exp,
            "param_grid": param_grid,
        }

        params = [params_sklearn, params_ackley]

        from hyperactive.utils.parallel import _get_parallel_test_fixtures

        parallel_fixtures = _get_parallel_test_fixtures()

        for x in parallel_fixtures:
            new_ackley = params_ackley.copy()
            new_ackley.update(x)
            params.append(new_ackley)

        return params
