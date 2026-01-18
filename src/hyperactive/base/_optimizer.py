"""Base class for optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from skbase.base import BaseObject


class BaseOptimizer(BaseObject):
    """Base class for optimizer."""

    _tags = {
        "object_type": "optimizer",
        "python_dependencies": None,
        # properties of the optimizer
        "info:name": None,  # str
        "info:local_vs_global": "mixed",  # "local", "mixed", "global"
        "info:explore_vs_exploit": "mixed",  # "explore", "exploit", "mixed"
        "info:compute": "middle",  # "low", "middle", "high"
        # see here for explanation of the tags:
        # https://simonblanke.github.io/gradient-free-optimizers-documentation/1.5/optimizers/  # noqa: E501
        # search space capabilities (conservative defaults)
        "capability:discrete": True,  # supports discrete lists
        "capability:continuous": False,  # supports continuous ranges
        "capability:categorical": True,  # supports categorical choices
        "capability:log_scale": False,  # supports log-scale sampling
        "capability:conditions": False,  # supports conditional params
        "capability:constraints": False,  # supports constraint functions
    }

    def __init__(self):
        super().__init__()
        assert hasattr(self, "experiment"), "Optimizer must have an experiment."
        search_config = self.get_params()
        self._experiment = search_config.pop("experiment", None)

        if self.get_tag("info:name") is None:
            self.set_tags(**{"info:name": self.__class__.__name__})

    def get_search_config(self):
        """Get the search configuration.

        Returns
        -------
        dict with str keys
            The search configuration dictionary.
        """
        search_config = self.get_params(deep=False)
        search_config.pop("experiment", None)
        return search_config

    def get_experiment(self):
        """Get the experiment.

        Returns
        -------
        BaseExperiment
            The experiment to optimize parameters for.
        """
        exp = self._experiment
        exp_is_baseobj = isinstance(exp, BaseObject)
        if not exp_is_baseobj or exp.get_tag("object_type") != "experiment":
            from hyperactive.experiment.func import FunctionExperiment

            exp = FunctionExperiment(exp)  # callable adapted to BaseExperiment
        return exp

    def solve(self):
        """Run the optimization search process to maximize the experiment's score.

        The optimization searches for a maximizer of the experiment's
        ``score`` method.

        Depending on the tag ``property:higher_or_lower_is_better`` being
        set to ``higher`` or ``lower``, the ``run`` method will search for:

        * the minimizer of the ``evaluate`` method if the tag is ``lower``
        * the maximizer of the ``evaluate`` method if the tag is ``higher``

        Returns
        -------
        best_params : dict
            The best parameters found during the optimization process.
            The dict ``best_params`` can be used in ``experiment.score`` or
            ``experiment.evaluate`` directly.
        """
        experiment = self.get_experiment()
        search_config = self.get_search_config()

        # Adapt search space for backend capabilities (e.g., categorical encoding)
        experiment, search_config, adapter = self._adapt_search_space(
            experiment, search_config
        )

        # Run optimization
        best_params = self._solve(experiment, **search_config)

        # Decode results if adapter was used
        if adapter is not None:
            best_params = adapter.decode(best_params)

        self.best_params_ = best_params
        return best_params

    def _adapt_search_space(self, experiment, search_config):
        """Adapt search space and experiment for backend capabilities.

        If the backend doesn't support certain search space features
        (e.g., categorical values), this method encodes the search space
        and wraps the experiment to handle encoding/decoding transparently.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize.
        search_config : dict
            The search configuration containing the search space.

        Returns
        -------
        experiment : BaseExperiment
            The experiment, possibly wrapped for decoding.
        search_config : dict
            The search config, possibly with encoded search space.
        adapter : SearchSpaceAdapter or None
            The adapter if encoding was applied, None otherwise.
        """
        from hyperactive.opt._adapters._search_space_adapter import SearchSpaceAdapter

        search_space_key = self._detect_search_space_key(search_config)

        # No search space found - pass through unchanged
        if not search_space_key or not search_config.get(search_space_key):
            return experiment, search_config, None

        # Create adapter with backend capabilities
        capabilities = {
            "categorical": self.get_tag("capability:categorical"),
            "continuous": self.get_tag("capability:continuous"),
        }
        adapter = SearchSpaceAdapter(search_config[search_space_key], capabilities)

        # Backend supports all features - pass through unchanged
        if not adapter.needs_encoding:
            return experiment, search_config, None

        # Encoding needed - transform search space and wrap experiment
        encoded_config = search_config.copy()
        encoded_config[search_space_key] = adapter.encode()
        wrapped_experiment = adapter.wrap_experiment(experiment)

        return wrapped_experiment, encoded_config, adapter

    def _detect_search_space_key(self, search_config):
        """Find which key holds the search space in the config.

        Parameters
        ----------
        search_config : dict
            The search configuration dictionary.

        Returns
        -------
        str or None
            The key name for search space, or None if not found.
        """
        for key in ["search_space", "param_space", "param_grid"]:
            if key in search_config and search_config[key] is not None:
                return key
        return None

    def _solve(self, experiment, *args, **kwargs):
        """Run the optimization search process.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        *args : tuple
            Positional arguments specific to the optimization backend.
        **kwargs : dict
            Keyword arguments specific to the optimization backend.

        Returns
        -------
        dict with str keys
            The best parameters found during the search.
            Must have keys a subset or identical to experiment.paramnames().
        """
        raise NotImplementedError(
            "abstract method, BaseOptimizer._solve should be implemented by "
            "descendant classes"
        )
