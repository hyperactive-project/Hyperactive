"""Base adapter for Optuna optimizers."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseOptunaAdapter"]


class _BaseOptunaAdapter(BaseOptimizer):
    """Base adapter for Optuna optimizers."""

    _tags = {
        "python_dependencies": ["optuna"],
        "info:name": "Optuna-based optimizer",
        # search space capabilities
        "capability:search_space:continuous": True,  # native support
        "capability:search_space:discrete": True,
        "capability:search_space:categorical": True,
        "capability:search_space:mixed": True,
        "capability:search_space:log_scale": True,  # native support
        "capability:search_space:conditional": True,  # via define-by-run
        "capability:search_space:constraints": False,  # not native
        "capability:search_space:distributions": True,  # via conversion
        "capability:search_space:nested": True,  # via conditional support
    }

    def __init__(
        self,
        param_space=None,
        n_trials=100,
        initialize=None,
        random_state=None,
        early_stopping=None,
        max_score=None,
        experiment=None,
        **optimizer_kwargs,
    ):
        self.param_space = param_space
        self.n_trials = n_trials
        self.initialize = initialize
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.max_score = max_score
        self.experiment = experiment
        self.optimizer_kwargs = optimizer_kwargs
        super().__init__()

    def _get_optimizer(self):
        """Get the Optuna optimizer to use.

        This method should be implemented by subclasses to return
        the specific optimizer class and its initialization parameters.

        Returns
        -------
        optimizer
            The Optuna optimizer instance
        """
        raise NotImplementedError("Subclasses must implement _get_optimizer")

    def _convert_param_space(self, param_space):
        """Convert parameter space to Optuna format.

        Parameters
        ----------
        param_space : dict or SearchSpace
            The parameter space to convert

        Returns
        -------
        dict
            The converted parameter space
        """
        from hyperactive.search_space import SearchSpace

        if isinstance(param_space, SearchSpace):
            # Validate SearchSpace features before conversion
            self._validate_search_space_features(param_space)
            # Store reference to original SearchSpace for constraints
            self._search_space_obj = param_space
            return param_space.to_backend("optuna")

        return param_space

    def _get_constraints_from_param_space(self):
        """Extract constraints from SearchSpace if available.

        Returns
        -------
        list or None
            List of constraint functions or None.
        """
        if hasattr(self, "_search_space_obj") and self._search_space_obj is not None:
            return [c.predicate for c in self._search_space_obj.constraints]
        return None

    def _suggest_params(self, trial, param_space):
        """Suggest parameters using Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object
        param_space : dict
            The parameter space

        Returns
        -------
        dict
            The suggested parameters
        """
        import optuna.distributions

        params = {}
        for key, space in param_space.items():
            # Check for Optuna distribution objects (e.g., FloatDistribution)
            if isinstance(space, optuna.distributions.BaseDistribution):
                params[key] = trial._suggest(key, space)
            elif isinstance(space, tuple) and len(space) == 2:
                # Tuples are treated as ranges (low, high)
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = trial.suggest_float(key, low, high, log=False)
            elif isinstance(space, list):
                # Lists are treated as categorical choices
                params[key] = trial.suggest_categorical(key, space)
            else:
                raise ValueError(f"Invalid parameter space for key '{key}': {space}")
        return params

    def _objective(self, trial):
        """Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object

        Returns
        -------
        float
            The objective value
        """
        params = self._suggest_params(trial, self.param_space)
        score = self.experiment(params)

        # Handle early stopping based on max_score
        if self.max_score is not None and score >= self.max_score:
            trial.study.stop()

        return score

    def _setup_initial_positions(self, study):
        """Set up initial starting positions if provided.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object
        """
        if self.initialize is not None:
            if isinstance(self.initialize, dict) and "warm_start" in self.initialize:
                warm_start_points = self.initialize["warm_start"]
                if isinstance(warm_start_points, list):
                    # For warm start, we manually add trials to the study history
                    # instead of using suggest methods to avoid distribution conflicts
                    for point in warm_start_points:
                        self.experiment(point)
                        study.enqueue_trial(point)

    def _solve(self, experiment, param_space, n_trials, **kwargs):
        """Run the Optuna optimization.

        Parameters
        ----------
        experiment : callable
            The experiment to optimize
        param_space : dict or SearchSpace
            The parameter space
        n_trials : int
            Number of trials
        **kwargs
            Additional parameters

        Returns
        -------
        dict
            The best parameters found
        """
        import optuna

        # Convert param_space (handles SearchSpace objects)
        self.param_space = self._convert_param_space(param_space)

        # Create optimizer with random state if provided
        optimizer = self._get_optimizer()

        # Create study
        study = optuna.create_study(
            direction="maximize",  # Assuming we want to maximize scores
            sampler=optimizer,
        )

        # Setup initial positions
        self._setup_initial_positions(study)

        # Setup early stopping callback
        callbacks = []
        if self.early_stopping is not None:

            def early_stopping_callback(study, trial):
                if len(study.trials) >= self.early_stopping:
                    study.stop()

            callbacks.append(early_stopping_callback)

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=n_trials,
            callbacks=callbacks if callbacks else None,
        )

        self.best_score_ = study.best_value
        self.best_params_ = study.best_params
        return study.best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from hyperactive.experiment.integrations import SklearnCvExperiment

        X, y = load_iris(return_X_y=True)
        sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

        param_space = {
            "C": (0.01, 10),
            "gamma": (0.0001, 10),
        }

        return [
            {
                "param_space": param_space,
                "n_trials": 10,
                "experiment": sklearn_exp,
            }
        ]
