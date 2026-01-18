"""Quasi-Monte Carlo optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from .._adapters._base_optuna_adapter import _BaseOptunaAdapter


class QMCOptimizer(_BaseOptunaAdapter):
    """Quasi-Monte Carlo optimizer.

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
    qmc_type : str, default="sobol"
        Type of QMC sequence. Options: "sobol", "halton".
    scramble : bool, default=True
        Whether to scramble the QMC sequence.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of QMCOptimizer with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import QMCOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = QMCOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.solve()
    """

    _tags = {
        "info:name": "Quasi-Monte Carlo Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "low",
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
        qmc_type="sobol",
        scramble=True,
        experiment=None,
    ):
        self.qmc_type = qmc_type
        self.scramble = scramble

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
        """Get the QMC optimizer.

        Returns
        -------
        optimizer
            The Optuna QMCOptimizer instance
        """
        import optuna

        optimizer_kwargs = {
            "qmc_type": self.qmc_type,
            "scramble": self.scramble,
        }

        if self.random_state is not None:
            optimizer_kwargs["seed"] = self.random_state

        return optuna.samplers.QMCSampler(**optimizer_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from hyperactive.experiment.integrations import SklearnCvExperiment

        # Test case 1: Halton sequence without scrambling
        params = super().get_test_params(parameter_set)
        params[0].update(
            {
                "qmc_type": "halton",
                "scramble": False,
            }
        )

        # Test case 2: Sobol sequence with scrambling
        X, y = load_iris(return_X_y=True)
        lr_exp = SklearnCvExperiment(
            estimator=LogisticRegression(random_state=42, max_iter=1000), X=X, y=y
        )

        mixed_param_space = {
            "C": (0.01, 100),  # Continuous
            "penalty": [
                "l1",
                "l2",
            ],  # Categorical - removed elasticnet to avoid solver conflicts
            "solver": ["liblinear", "saga"],  # Categorical
        }

        params.append(
            {
                "param_space": mixed_param_space,
                "n_trials": 16,  # Power of 2 for better QMC properties
                "experiment": lr_exp,
                "qmc_type": "sobol",  # Different sequence type
                "scramble": True,  # With scrambling for randomization
            }
        )

        # Test case 3: Different sampler configuration with same experiment
        params.append(
            {
                "param_space": mixed_param_space,
                "n_trials": 8,  # Power of 2, good for QMC
                "experiment": lr_exp,
                "qmc_type": "halton",  # Different QMC type
                "scramble": False,
            }
        )

        return params
