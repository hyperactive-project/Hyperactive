"""Grid optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class GridOptimizer(_BaseOptunaAdapter):
    """Grid search optimizer.

    Parameters
    ----------
    param_space : dict[str, tuple or list or optuna distributions]
        The search space to explore. Dictionary with parameter names
        as keys and either tuples/lists of (low, high) or
        optuna distribution objects as values.
    n_trials : int, default=100
        Number of optimization trials.
    initialize : dict[str, int], default=None
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    random_state : None, int, default=None
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    early_stopping : int, default=None
        Number of trials after which to stop if no improvement.
    max_score : float, default=None
        Maximum score threshold. Stop optimization when reached.
    search_space : dict, default=None
        Explicit search space for grid search.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of GridOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import GridOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": [0.01, 0.1, 1, 10],
    ...     "gamma": [0.0001, 0.01, 0.1, 1],
    ... }
    >>> optimizer = GridOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "Grid Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "low",
        "python_dependencies": ["optuna"],
        # Grid search needs discrete values, cannot sample from continuous ranges
        "capability:continuous": False,
    }

    def __init__(
        self,
        unified_space=None,
        param_space=None,
        n_trials=100,
        initialize=None,
        random_state=None,
        early_stopping=None,
        max_score=None,
        experiment=None,
    ):
        super().__init__(
            unified_space=unified_space,
            param_space=param_space,
            n_trials=n_trials,
            initialize=initialize,
            random_state=random_state,
            early_stopping=early_stopping,
            max_score=max_score,
            experiment=experiment,
        )

    def _get_optimizer(self):
        """Get the Grid optimizer.

        Returns
        -------
        optimizer
            The Optuna GridOptimizer instance
        """
        import optuna

        # Convert param_space to Optuna grid search space format
        # Use _resolved_param_space which is set in _solve before this is called
        search_space = {}
        param_space = getattr(self, "_resolved_param_space", self.param_space)

        if param_space is not None:
            for key, space in param_space.items():
                if isinstance(space, list):
                    search_space[key] = space
                elif isinstance(space, tuple) and len(space) == 2:
                    # Convert range to discrete list for grid search
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        search_space[key] = list(range(low, high + 1))
                    else:
                        # Create a reasonable grid for continuous spaces
                        import numpy as np

                        search_space[key] = np.linspace(low, high, 10).tolist()

        return optuna.samplers.GridSampler(search_space)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        from hyperactive.experiment.integrations import SklearnCvExperiment

        X, y = load_iris(return_X_y=True)

        # Test case 1: Basic continuous parameters (converted to discrete)
        svm_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
        param_space_1 = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.0001, 0.01, 0.1, 1],
        }

        # Test case 2: Mixed categorical and discrete parameters
        knn_exp = SklearnCvExperiment(estimator=KNeighborsClassifier(), X=X, y=y)
        param_space_2 = {
            "n_neighbors": [1, 3, 5, 7],  # Discrete integers
            "weights": ["uniform", "distance"],  # Categorical
            "metric": ["euclidean", "manhattan"],  # Categorical
            "p": [1, 2],  # Discrete for minkowski
        }

        # Test case 3: Small exhaustive grid (tests complete enumeration)
        param_space_3 = {
            "C": [0.1, 1],  # 2 values
            "kernel": ["rbf", "linear"],  # 2 values
        }
        # Total: 2 x 2 = 4 combinations, n_trials should cover all

        return [
            {
                "param_space": param_space_1,
                "n_trials": 10,
                "experiment": svm_exp,
            },
            {
                "param_space": param_space_2,
                "n_trials": 15,
                "experiment": knn_exp,
            },
            {
                "param_space": param_space_3,
                "n_trials": 4,  # Exact number for exhaustive search
                "experiment": svm_exp,
            },
        ]
