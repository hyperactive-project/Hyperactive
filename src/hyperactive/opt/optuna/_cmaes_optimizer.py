"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class CmaEsOptimizer(_BaseOptunaAdapter):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

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
    x0 : dict, default=None
        Initial parameter values for CMA-ES.
    sigma0 : float, default=1.0
        Initial standard deviation for CMA-ES.
    n_startup_trials : int, default=1
        Number of startup trials for CMA-ES.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of CmaEsOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import CmaEsOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = CmaEsOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "CMA-ES Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
        "info:compute": "high",
        "python_dependencies": ["optuna", "cmaes"],
    }

    def __init__(
        self,
        param_space=None,
        n_trials=100,
        initialize=None,
        random_state=None,
        early_stopping=None,
        max_score=None,
        x0=None,
        sigma0=1.0,
        n_startup_trials=1,
        experiment=None,
    ):
        self.x0 = x0
        self.sigma0 = sigma0
        self.n_startup_trials = n_startup_trials

        super().__init__(
            param_space=param_space,
            n_trials=n_trials,
            initialize=initialize,
            random_state=random_state,
            early_stopping=early_stopping,
            max_score=max_score,
            experiment=experiment,
        )

    def _get_optimizer(self):
        """Get the CMA-ES optimizer.

        Returns
        -------
        optimizer
            The Optuna CmaEsOptimizer instance
        """
        import optuna

        try:
            import cmaes  # noqa: F401
        except ImportError:
            raise ImportError(
                "CmaEsOptimizer requires the 'cmaes' package. "
                "Install it with: pip install cmaes"
            )

        optimizer_kwargs = {
            "sigma0": self.sigma0,
            "n_startup_trials": self.n_startup_trials,
        }

        if self.x0 is not None:
            optimizer_kwargs["x0"] = self.x0

        if self.random_state is not None:
            optimizer_kwargs["seed"] = self.random_state

        return optuna.samplers.CmaEsSampler(**optimizer_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from sklearn.datasets import make_regression
        from sklearn.neural_network import MLPRegressor

        from hyperactive.experiment.bench import Sphere
        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Test case 1: Basic continuous parameters (from base)
        params = super().get_test_params(parameter_set)
        params[0].update(
            {
                "sigma0": 0.5,
                "n_startup_trials": 1,
            }
        )

        # Test case 2: Neural network with continuous parameters only
        # (CMA-ES specific - only continuous parameters allowed)
        X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
        mlp_exp = SklearnCvExperiment(
            estimator=MLPRegressor(random_state=42, max_iter=100), X=X, y=y, cv=3
        )

        continuous_param_space = {
            "alpha": (1e-5, 1e-1),  # L2 regularization (continuous)
            "learning_rate_init": (1e-4, 1e-1),  # Learning rate (continuous)
            "beta_1": (0.8, 0.99),  # Adam beta1 (continuous)
            "beta_2": (0.9, 0.999),  # Adam beta2 (continuous)
            # Note: No categorical parameters - CMA-ES doesn't support them
        }

        params.append(
            {
                "param_space": continuous_param_space,
                "n_trials": 8,  # Smaller for faster testing
                "experiment": mlp_exp,
                "sigma0": 0.3,  # Different sigma for diversity
                "n_startup_trials": 2,  # More startup trials
            }
        )

        # Test case 3: High-dimensional continuous space (CMA-ES strength)
        # Use Sphere benchmark which expects x0, x1, ..., xN params
        sphere_exp = Sphere(n_dim=6)
        high_dim_continuous = {
            f"x{i}": (-1.0, 1.0)
            for i in range(6)  # 6D continuous optimization
        }

        params.append(
            {
                "param_space": high_dim_continuous,
                "n_trials": 12,
                "experiment": sphere_exp,
                "sigma0": 0.7,  # Larger initial spread
                "n_startup_trials": 3,
            }
        )

        return params
