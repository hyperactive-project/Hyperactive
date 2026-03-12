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
        from hyperactive.opt._adapters._adapter_utils import adapt_search_space

        experiment = self.get_experiment()
        search_config = self.get_search_config()

        # Adapt search space for backend capabilities (e.g., categorical encoding)
        capabilities = {
            "categorical": self.get_tag("capability:categorical"),
            "continuous": self.get_tag("capability:continuous"),
        }
        experiment, search_config, adapter = adapt_search_space(
            experiment, search_config, capabilities
        )

        # Run optimization
        best_params = self._solve(experiment, **search_config)

        # Decode results if adapter was used
        if adapter is not None:
            best_params = adapter.decode(best_params)

        self.best_params_ = best_params
        return best_params

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
