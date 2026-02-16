"""Random search optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import ParameterSampler

from hyperactive.base import BaseOptimizer
from hyperactive.opt._common import _score_params
from hyperactive.utils.parallel import parallelize


class RandomSearchSk(BaseOptimizer):
    """Random search optimizer leveraging sklearn's ``ParameterSampler``.

    Parameters
    ----------
    unified_space : dict[str, list], default=None
        Unified search space format: dict mapping parameter names to lists.
        Use this for portable search space definitions across backends.
        Mutually exclusive with ``param_distributions``.
    param_distributions : dict[str, list | scipy.stats.rv_frozen], default=None
        Search space specification. Discrete lists are sampled uniformly;
        scipy distribution objects are sampled via their ``rvs`` method.

    n_iter : int, default=10
        Number of parameter sets to evaluate.

    random_state : int | np.random.RandomState | None, default=None
        Controls the pseudo-random generator for reproducibility.

    error_score : float, default=np.nan
        Score assigned when the experiment raises an exception.

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
        Callable returning a scalar score when invoked with keyword
        arguments matching a parameter set.

    Example
    -------
    Random search with different backend configurations:

    >>> from hyperactive.opt import RandomSearchSk
    >>> from scipy.stats import uniform
    >>> param_distributions = {
    ...     "C": uniform(loc=0.1, scale=10),
    ...     "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    ... }
    >>>
    >>> # Sequential execution
    >>> random_search = RandomSearchSk(
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     backend="None",
    ... )
    >>>
    >>> # Parallel execution with threading backend
    >>> random_search_parallel = RandomSearchSk(
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     backend="threading",
    ...     backend_params={"n_jobs": 2},
    ... )

    Attributes
    ----------
    best_params_ : dict[str, Any]
        Hyper-parameter configuration with the best (lowest) score.
    best_score_ : float
        Score achieved by ``best_params_``.
    best_index_ : int
        Index of ``best_params_`` in the sampled sequence.
    """

    def __init__(
        self,
        unified_space=None,
        param_distributions=None,
        n_iter=10,
        random_state=None,
        error_score=np.nan,
        backend="None",
        backend_params=None,
        experiment=None,
    ):
        self.experiment = experiment
        self.unified_space = unified_space
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
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

        # Resolve: unified_space is converted to param_distributions
        unified_space = search_config.pop("unified_space", None)
        param_distributions = search_config.get("param_distributions")

        # Validate: only one should be set
        if unified_space is not None and param_distributions is not None:
            raise ValueError(
                "Provide either 'unified_space' or 'param_distributions', not both. "
                "Use 'unified_space' for simple dict[str, list] format, "
                "or 'param_distributions' for native sklearn format with distributions."
            )

        # Use unified_space if param_distributions is not set
        if unified_space is not None:
            search_config["param_distributions"] = unified_space

        return search_config

    @staticmethod
    def _is_distribution(obj) -> bool:
        """Return True if *obj* looks like a scipy frozen distribution."""
        return callable(getattr(obj, "rvs", None))

    def _check_param_distributions(self, param_distributions):
        """Validate ``param_distributions`` similar to sklearn ≤1.0.x."""
        if hasattr(param_distributions, "items"):
            param_distributions = [param_distributions]

        for p in param_distributions:
            for name, v in p.items():
                if self._is_distribution(v):
                    # Assume scipy frozen distribution: nothing to check
                    continue

                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, np.ndarray | Sequence):
                    raise ValueError(
                        f"Parameter distribution for ({name}) must be a list, numpy "
                        f"array, or scipy.stats ``rv_frozen``, but got ({type(v)})."
                        " Single values need to be wrapped in a sequence."
                    )

                if len(v) == 0:
                    raise ValueError(
                        f"Parameter values for ({name}) need to be a "
                        "non-empty sequence."
                    )

    def _solve(
        self,
        experiment,
        param_distributions,
        n_iter,
        random_state,
        error_score,
        backend,
        backend_params,
    ):
        """Sample ``n_iter`` points and return the best parameter set."""
        self._check_param_distributions(param_distributions)

        sampler = ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state,
        )
        candidate_params = list(sampler)

        meta = {
            "experiment": experiment,
            "error_score": error_score,
        }

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

        # public attributes for external consumers (signed score convention)
        self.best_index_ = best_index
        self.best_score_ = float(scores[best_index])
        self.best_params_ = best_params

        return best_params

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Provide deterministic benchmark configurations for unit tests."""
        from hyperactive.experiment.bench import Ackley
        from hyperactive.experiment.integrations import SklearnCvExperiment

        # 1) ML example (Iris + SVC)
        sklearn_exp = SklearnCvExperiment.create_test_instance()
        param_dist_1 = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": np.logspace(-4, 1, 6),
        }
        params_sklearn = {
            "experiment": sklearn_exp,
            "param_distributions": param_dist_1,
            "n_iter": 5,
            "random_state": 42,
        }

        # 2) continuous optimisation example (Ackley)
        ackley_exp = Ackley.create_test_instance()
        param_dist_2 = {
            "x0": np.linspace(-5, 5, 50),
            "x1": np.linspace(-5, 5, 50),
        }
        params_ackley = {
            "experiment": ackley_exp,
            "param_distributions": param_dist_2,
            "n_iter": 20,
            "random_state": 0,
        }

        params = [params_sklearn, params_ackley]

        from hyperactive.utils.parallel import _get_parallel_test_fixtures

        parallel_fixtures = _get_parallel_test_fixtures()

        for x in parallel_fixtures:
            new_ackley = params_ackley.copy()
            new_ackley.update(x)
            params.append(new_ackley)

        return params
