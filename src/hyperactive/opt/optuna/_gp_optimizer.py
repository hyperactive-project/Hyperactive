"""Gaussian Process optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class GPOptimizer(_BaseOptunaAdapter):
    """Gaussian Process-based Bayesian optimizer.

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
    n_startup_trials : int, default=10
        Number of startup trials for GP.
    deterministic_objective : bool, default=False
        Whether the objective function is deterministic.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of GPOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import GPOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = GPOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "Gaussian Process Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "exploit",
        "info:compute": "high",
        "python_dependencies": ["optuna", "torch"],
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
        n_startup_trials=10,
        deterministic_objective=False,
        experiment=None,
    ):
        self.n_startup_trials = n_startup_trials
        self.deterministic_objective = deterministic_objective

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
        """Get the GP optimizer.

        Returns
        -------
        optimizer
            The Optuna GPOptimizer instance
        """
        import optuna

        optimizer_kwargs = {
            "n_startup_trials": self.n_startup_trials,
            "deterministic_objective": self.deterministic_objective,
        }

        if self.random_state is not None:
            optimizer_kwargs["seed"] = self.random_state

        return optuna.samplers.GPSampler(**optimizer_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        params = super().get_test_params(parameter_set)
        params[0].update(
            {
                "n_startup_trials": 5,
                "deterministic_objective": True,
            }
        )
        return params
