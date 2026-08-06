"""Hyperband optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import math
import random

import numpy as np

from hyperactive.base import BaseOptimizer
from hyperactive.opt._common import _score_params


class Hyperband(BaseOptimizer):
    """Hyperband optimizer for multi-fidelity hyperparameter optimization.

    Hyperband improves efficiency by dynamically allocating computational
    resources. It is based on the Successive Halving Algorithm (SHA) but
    introduces an adaptive mechanism to balance exploration (trying many
    configurations) and exploitation (allocating resources to promising
    candidates).

    The algorithm runs multiple brackets of Successive Halving with different
    exploration-exploitation trade-offs. Early brackets are more aggressive
    (many configs, low initial resource) while later brackets are more
    conservative (fewer configs, high initial resource).

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter names as
        keys and lists of possible values. Must include the resource parameter.
    resource_name : str
        Name of the parameter in ``search_space`` that represents the
        resource (e.g., ``"n_epochs"``, ``"n_estimators"``). This parameter
        is controlled internally by Hyperband.
    max_resource : int, default=81
        Maximum amount of resource that can be allocated to a single
        configuration. Should be a power of ``eta`` for clean bracket
        divisions.
    eta : int, default=3
        The proportion of configurations discarded in each round of
        Successive Halving. Configurations are reduced by a factor of
        ``eta`` after each round.
    random_state : int or None, default=None
        Seed for reproducibility. If None, results are non-deterministic.
    error_score : float, default=np.nan
        Score assigned when the experiment raises an exception.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Attributes
    ----------
    best_params_ : dict
        The best parameters found during the optimization.

    References
    ----------
    .. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A.
       (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter
       Optimization. JMLR, 18(185), 1-52.

    Examples
    --------
    Hyperband applied to a simple function optimization:

    >>> import numpy as np
    >>> from hyperactive.opt import Hyperband

    1. defining the experiment:
    >>> def objective(params):
    ...     return -(params["x"] ** 2 + params["y"] ** 2)

    2. setting up the Hyperband optimizer:
    >>> hb = Hyperband(
    ...     search_space={
    ...         "x": list(np.arange(-5, 5, 1.0)),
    ...         "y": list(np.arange(-5, 5, 1.0)),
    ...         "resource": [1, 3, 9, 27, 81],
    ...     },
    ...     resource_name="resource",
    ...     max_resource=81,
    ...     eta=3,
    ...     random_state=42,
    ...     experiment=objective,
    ... )

    3. running the Hyperband optimizer:
    >>> best_params = hb.solve()
    >>> isinstance(best_params, dict)
    True
    """

    _tags = {
        "info:name": "Hyperband",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
        "info:compute": "middle",
    }

    def __init__(
        self,
        search_space=None,
        resource_name="resource",
        max_resource=81,
        eta=3,
        random_state=None,
        error_score=np.nan,
        experiment=None,
    ):
        self.search_space = search_space
        self.resource_name = resource_name
        self.max_resource = max_resource
        self.eta = eta
        self.random_state = random_state
        self.error_score = error_score
        self.experiment = experiment

        super().__init__()

    def _sample_configuration(self, rng, hp_space):
        """Sample a random configuration from the hyperparameter search space.

        Parameters
        ----------
        rng : random.Random
            Random number generator.
        hp_space : dict
            Search space excluding the resource parameter.

        Returns
        -------
        dict
            A randomly sampled parameter configuration.
        """
        config = {}
        for name, values in hp_space.items():
            config[name] = rng.choice(values)
        return config

    def _solve(
        self,
        experiment,
        search_space,
        resource_name,
        max_resource,
        eta,
        random_state,
        error_score,
    ):
        """Run the Hyperband optimization process.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_space : dict
            The search space to explore.
        resource_name : str
            Name of the resource parameter.
        max_resource : int
            Maximum resource per configuration.
        eta : int
            Reduction factor.
        random_state : int or None
            Random state for reproducibility.
        error_score : float
            Score for failed evaluations.

        Returns
        -------
        dict
            The best parameters found during the search.
        """
        rng = random.Random(random_state)  # noqa: S311

        hp_space = {k: v for k, v in search_space.items() if k != resource_name}

        s_max = int(math.floor(math.log(max_resource, eta)))
        B = (s_max + 1) * max_resource  # noqa: N806

        best_score = -np.inf
        best_params = None

        meta = {
            "experiment": experiment,
            "error_score": error_score,
        }

        # outer loop: iterate over brackets
        for s in range(s_max, -1, -1):
            # n = initial number of configs for this bracket
            n = int(math.ceil((B / max_resource) * (eta**s / (s + 1))))

            # sample n random configurations
            configs = [self._sample_configuration(rng, hp_space) for _ in range(n)]

            # inner loop: Successive Halving
            for i in range(s + 1):
                n_i = int(math.floor(n * eta ** (-i)))

                # evaluate configs at this round
                scores = []
                for config in configs:
                    # only pass non-resource params to experiment
                    score = _score_params(config, meta)
                    scores.append(score)

                    # track global best (without resource)
                    if score > best_score:
                        best_score = score
                        best_params = config.copy()

                # keep top 1/eta fraction
                n_keep = max(int(math.floor(n_i / eta)), 1)
                indices = np.argsort(scores)[::-1][:n_keep]
                configs = [configs[idx] for idx in indices]

        self.best_params_ = best_params
        return best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        import numpy as np

        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()

        params_ackley = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": list(np.linspace(-5, 5, 10)),
                "x1": list(np.linspace(-5, 5, 10)),
                "resource": [1, 3, 9, 27],
            },
            "resource_name": "resource",
            "max_resource": 27,
            "eta": 3,
            "random_state": 42,
        }

        params_small = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": list(np.linspace(-2, 2, 5)),
                "x1": list(np.linspace(-2, 2, 5)),
                "resource": [1, 3, 9],
            },
            "resource_name": "resource",
            "max_resource": 9,
            "eta": 3,
            "random_state": 0,
        }

        return [params_ackley, params_small]
