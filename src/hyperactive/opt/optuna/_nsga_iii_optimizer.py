"""NSGA-III multi-objective optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class NSGAIIIOptimizer(_BaseOptunaAdapter):
    """NSGA-III multi-objective optimizer.

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
    population_size : int, default=50
        Population size for NSGA-III.
    mutation_prob : float, default=0.1
        Mutation probability for NSGA-III.
    crossover_prob : float, default=0.9
        Crossover probability for NSGA-III.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of NSGAIIIOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import NSGAIIIOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = NSGAIIIOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "NSGA-III Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
        "info:compute": "high",
        "python_dependencies": ["optuna"],
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
        population_size=50,
        mutation_prob=0.1,
        crossover_prob=0.9,
        experiment=None,
    ):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

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
        """Get the NSGA-III optimizer.

        Returns
        -------
        optimizer
            The Optuna NSGAIIIOptimizer instance
        """
        import optuna

        optimizer_kwargs = {
            "population_size": self.population_size,
            "mutation_prob": self.mutation_prob,
            "crossover_prob": self.crossover_prob,
        }

        if self.random_state is not None:
            optimizer_kwargs["seed"] = self.random_state

        return optuna.samplers.NSGAIIISampler(**optimizer_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        params = super().get_test_params(parameter_set)
        params[0].update(
            {
                "population_size": 20,
                "mutation_prob": 0.2,
                "crossover_prob": 0.8,
            }
        )
        return params
