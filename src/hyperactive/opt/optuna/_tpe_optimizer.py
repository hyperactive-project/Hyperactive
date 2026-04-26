"""TPE (Tree-structured Parzen Estimator) optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class TPEOptimizer(_BaseOptunaAdapter):
    """Tree-structured Parzen Estimator optimizer.

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
        Number of startup trials for TPE.
    n_ei_candidates : int, default=24
        Number of candidates for expected improvement.
    weights : callable, default=None
        Weight function for TPE.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of TPEOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import TPEOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = TPEOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "Tree-structured Parzen Estimator",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "exploit",
        "info:compute": "middle",
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
        n_startup_trials=10,
        n_ei_candidates=24,
        weights=None,
        experiment=None,
    ):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.weights = weights

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
        """Get the TPE optimizer.

        Returns
        -------
        optimizer
            The Optuna TPEOptimizer instance
        """
        import optuna

        optimizer_kwargs = {
            "n_startup_trials": self.n_startup_trials,
            "n_ei_candidates": self.n_ei_candidates,
        }

        if self.weights is not None:
            optimizer_kwargs["weights"] = self.weights

        if self.random_state is not None:
            optimizer_kwargs["seed"] = self.random_state

        return optuna.samplers.TPESampler(**optimizer_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Test case 1: Basic TPE with standard parameters
        params = super().get_test_params(parameter_set)
        params[0].update(
            {
                "n_startup_trials": 5,
                "n_ei_candidates": 12,
            }
        )

        # Test case 2: Mixed parameter types with warm start
        X, y = load_wine(return_X_y=True)
        rf_exp = SklearnCvExperiment(
            estimator=RandomForestClassifier(random_state=42), X=X, y=y
        )

        mixed_param_space = {
            "n_estimators": (10, 100),  # Continuous integer
            "max_depth": [3, 5, 7, 10, None],  # Mixed discrete/None
            "criterion": ["gini", "entropy"],  # Categorical
            "min_samples_split": (2, 20),  # Continuous integer
            "bootstrap": [True, False],  # Boolean
        }

        # Warm start with known good configuration
        warm_start_points = [
            {
                "n_estimators": 50,
                "max_depth": 5,
                "criterion": "gini",
                "min_samples_split": 2,
                "bootstrap": True,
            }
        ]

        params.append(
            {
                "param_space": mixed_param_space,
                "n_trials": 20,
                "experiment": rf_exp,
                "n_startup_trials": 3,  # Fewer random trials before TPE
                "n_ei_candidates": 24,  # More EI candidates for better optimization
                "initialize": {"warm_start": warm_start_points},
            }
        )

        # Test case 3: High-dimensional continuous space (TPE strength)
        svm_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
        high_dim_space = {
            "C": (0.01, 100),
            "gamma": (1e-6, 1e2),
            "coef0": (0.0, 10.0),
            "degree": (2, 5),
            "tol": (1e-5, 1e-2),
        }

        params.append(
            {
                "param_space": high_dim_space,
                "n_trials": 25,
                "experiment": svm_exp,
                "n_startup_trials": 8,  # More startup for exploration
                "n_ei_candidates": 32,  # More candidates for complex space
            }
        )

        return params
