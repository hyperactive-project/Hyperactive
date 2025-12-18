"""Auto optimizer that automatically selects the best algorithm.

This module provides the AutoOptimizer class that acts like any other
optimizer but automatically selects the best algorithm for the problem.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from typing import Optional, Type

from hyperactive.base import BaseOptimizer

from .selector import AlgorithmSelector


class AutoOptimizer(BaseOptimizer):
    """Optimizer that automatically selects the best algorithm.

    AutoOptimizer analyzes the objective function and search space to
    automatically select and run the most suitable optimization algorithm.
    It can be used as a drop-in replacement for any specific optimizer.

    Parameters
    ----------
    experiment : BaseExperiment or callable
        The experiment or objective function to optimize.
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and lists of possible values.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    random_state : int or None, default=None
        Random state for reproducibility.
    expand_source : bool, default=True
        Whether to expand function source code when analyzing.
    verbose : bool, default=False
        If True, print information about the selected algorithm.

    Attributes
    ----------
    selected_optimizer_ : type
        The optimizer class that was selected.
    selection_scores_ : dict
        Dictionary mapping optimizer classes to their selection scores.
    optimizer_instance_ : BaseOptimizer
        The actual optimizer instance used.
    best_params_ : dict
        The best parameters found during optimization.

    Examples
    --------
    >>> from hyperactive.algorithm_selection import AutoOptimizer
    >>> def objective(x):
    ...     return -(x["a"] ** 2 + x["b"] ** 2)  # Simple quadratic
    >>> search_space = {"a": list(range(-5, 6)), "b": list(range(-5, 6))}
    >>> auto = AutoOptimizer(
    ...     experiment=objective,
    ...     search_space=search_space,
    ...     n_iter=50,
    ... )
    >>> best_params = auto.solve()
    >>> auto.selected_optimizer_.__name__  # See which algorithm was chosen
    'HillClimbing'
    """

    _tags = {
        "info:name": "Auto Optimizer",
        "info:local_vs_global": "mixed",
        "info:explore_vs_exploit": "mixed",
        "info:compute": "middle",
    }

    def __init__(
        self,
        search_space=None,
        n_iter=100,
        random_state=None,
        expand_source=True,
        verbose=False,
        experiment=None,
    ):
        self.search_space = search_space
        self.n_iter = n_iter
        self.random_state = random_state
        self.expand_source = expand_source
        self.verbose = verbose
        self.experiment = experiment

        super().__init__()

        # Attributes set after solve()
        self.selected_optimizer_: Optional[Type[BaseOptimizer]] = None
        self.selection_scores_: Optional[dict] = None
        self.optimizer_instance_: Optional[BaseOptimizer] = None

    def _solve(self, experiment, **search_config):
        """Run the optimization with automatic algorithm selection.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        **search_config : dict
            Search configuration parameters.

        Returns
        -------
        dict
            The best parameters found.
        """
        search_space = search_config.get("search_space", self.search_space)
        n_iter = search_config.get("n_iter", self.n_iter)
        random_state = search_config.get("random_state", self.random_state)

        if search_space is None:
            raise ValueError("search_space must be provided")

        # Get the objective function from experiment
        objective_func = self._get_objective_func(experiment)

        # Select the best algorithm
        selector = AlgorithmSelector(expand_source=self.expand_source)
        self.selection_scores_ = selector.rank(objective_func, search_space, n_iter)
        self.selected_optimizer_ = selector.recommend(
            objective_func, search_space, n_iter
        )

        if self.verbose:
            print(f"AutoOptimizer selected: {self.selected_optimizer_.__name__}")
            print("Top 3 candidates:")
            for i, (opt_class, score) in enumerate(
                list(self.selection_scores_.items())[:3]
            ):
                print(f"  {i+1}. {opt_class.__name__}: {score:.3f}")

        # Build the optimizer config
        optimizer_config = {
            "experiment": experiment,
            "search_space": search_space,
            "n_iter": n_iter,
        }

        # Add random_state if supported
        if random_state is not None:
            optimizer_config["random_state"] = random_state

        # Instantiate and run the selected optimizer
        self.optimizer_instance_ = self.selected_optimizer_(**optimizer_config)
        best_params = self.optimizer_instance_.solve()

        return best_params

    def _get_objective_func(self, experiment):
        """Extract the objective function from an experiment.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment object.

        Returns
        -------
        callable
            The objective function.
        """
        from hyperactive.experiment.func import FunctionExperiment

        if isinstance(experiment, FunctionExperiment):
            return experiment.func
        elif hasattr(experiment, "_evaluate"):
            # For other experiment types, use the evaluate method
            return experiment._evaluate
        else:
            # Assume it's a callable
            return experiment

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        import numpy as np

        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()
        params = {
            "search_space": {
                "x0": list(np.linspace(-5, 5, 10)),
                "x1": list(np.linspace(-5, 5, 10)),
            },
            "n_iter": 10,
            "experiment": ackley_exp,
        }
        return params
