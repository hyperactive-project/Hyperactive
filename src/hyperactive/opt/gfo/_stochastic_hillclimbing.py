"""Hill climbing optimizer from gfo."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class StochasticHillClimbing(_BaseGFOadapter):
    """Stochastic hill climbing optimizer.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
        Optional, can be passed later via ``set_params``.
    initialize : dict[str, int], default={"grid": 4, "random": 2, "vertices": 4}
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable], default=[]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int, default=None
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float, default=0.1
        The probability of a random iteration during the the search process.
    epsilon : float, default=0.01
        The step-size for the climbing.
    distribution : str, default="normal"
        The type of distribution to sample from.
    n_neighbours : int, default=10
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    p_accept : float, default=0.5
        The probability of accepting a transition in the hill climbing process.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    verbose : bool, default=False
        If True, print the progress of the optimization process.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Hill climbing applied to scikit-learn parameter tuning:

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

    2. setting up the hill climbing optimizer:
    >>> from hyperactive.opt import StochasticHillClimbing
    >>> import numpy as np
    >>>
    >>> config = {
    ...     "search_space": {
    ...         "C": [0.01, 0.1, 1, 10],
    ...         "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ...     },
    ...     "n_iter": 100,
    ... }
    >>> hillclimbing = StochasticHillClimbing(experiment=sklearn_exp, **config)

    3. running the hill climbing search:
    >>> best_params = hillclimbing.solve()

    Best parameters can also be accessed via the attributes:
    >>> best_params = hillclimbing.best_params_
    """

    _tags = {
        "info:name": "Hill Climbing",
        "info:local_vs_global": "local",  # "local", "mixed", "global"
        "info:explore_vs_exploit": "exploit",  # "explore", "exploit", "mixed"
        "info:compute": "low",  # "low", "middle", "high"
    }

    def __init__(
        self,
        unified_space=None,
        search_space=None,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0.1,
        epsilon=0.01,
        distribution="normal",
        n_neighbours=10,
        p_accept=0.5,
        n_iter=100,
        verbose=False,
        experiment=None,
    ):
        self.unified_space = unified_space
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
        self.p_accept = p_accept
        self.n_iter = n_iter
        self.experiment = experiment
        self.verbose = verbose

        super().__init__()

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        from gradient_free_optimizers import StochasticHillClimbingOptimizer

        return StochasticHillClimbingOptimizer

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get the test parameters for the optimizer.

        Returns
        -------
        dict with str keys
            The test parameters dictionary.
        """
        params = super().get_test_params()
        experiment = params[0]["experiment"]
        more_params = {
            "experiment": experiment,
            "p_accept": 0.33,
            "search_space": {
                "C": [0.01, 0.1, 1, 10],
                "gamma": [0.0001, 0.01, 0.1, 1, 10],
            },
            "n_iter": 100,
        }
        params.append(more_params)
        return params
