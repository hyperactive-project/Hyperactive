from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class DownhillSimplexOptimizer(_BaseGFOadapter):
    """Downhill simplex optimizer.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    alpha : float
        The reflection parameter of the simplex algorithm.
    gamma : float
        The expansion parameter of the simplex algorithm.
    beta : float
        The contraction parameter of the simplex algorithm.
    sigma : float
        The shrinking parameter of the simplex algorithm.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    verbose : bool, default=False
        If True, print the progress of the optimization process.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of DownhillSimplexOptimizer with a scikit-learn experiment:

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

    2. setting up the downhillSimplexOptimizer optimizer:
    >>> from hyperactive.opt import DownhillSimplexOptimizer
    >>> import numpy as np
    >>>
    >>> config = {
    ...     "search_space": {
    ...         "C": [0.01, 0.1, 1, 10],
    ...         "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ...     },
    ...     "n_iter": 100,
    ... }
    >>> optimizer = DownhillSimplexOptimizer(experiment=sklearn_exp, **config)

    3. running the optimization:
    >>> best_params = optimizer.solve()

    Best parameters can also be accessed via:
    >>> best_params = optimizer.best_params_
    """

    _tags = {
        "info:name": "Downhill Simplex",
        "info:local_vs_global": "local",
        "info:explore_vs_exploit": "exploit",
        "info:compute": "low",
    }

    def __init__(
        self,
        unified_space=None,
        search_space=None,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0.1,
        alpha=1,
        gamma=2,
        beta=0.5,
        sigma=0.5,
        n_iter=100,
        verbose=False,
        experiment=None,
    ):
        self.unified_space = unified_space
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
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
        from gradient_free_optimizers import DownhillSimplexOptimizer

        return DownhillSimplexOptimizer

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
            "alpha": 0.33,
            "beta": 0.33,
            "gamma": 0.33,
            "sigma": 0.33,
            "search_space": {
                "C": [0.01, 0.1, 1, 10],
                "gamma": [0.0001, 0.01, 0.1, 1, 10],
            },
            "n_iter": 100,
        }
        params.append(more_params)
        return params
