"""Base class for Surfaces test function wrappers."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseExperiment


class BaseSurfacesExperiment(BaseExperiment):
    """Base class for wrapping Surfaces test functions as Hyperactive experiments.

    This adapter class wraps test functions from the Surfaces package,
    making them compatible with Hyperactive's experiment interface.
    Each subclass wraps a specific Surfaces test function.

    The wrapped Surfaces function is instantiated during ``__init__`` and
    stored in ``_surfaces_func``. The ``_evaluate`` method delegates to
    this wrapped function.

    Subclasses must set:
        _surfaces_class : class
            The Surfaces test function class to wrap.

    Subclasses should override:
        _tags : dict
            Tags specific to the wrapped function (dimensionality, modality, etc.).

    Parameters
    ----------
    metric : str, default="loss"
        Metric mode for the wrapped Surfaces function.
        For mathematical functions: "loss" (minimize) or "score" (maximize).
        For ML functions: scoring metric like "accuracy", "r2", etc.

    See Also
    --------
    surfaces.test_functions.BaseTestFunction :
        The base class in Surfaces that this adapter wraps.

    Notes
    -----
    The Surfaces package must be installed to use these experiments.
    Install with: ``pip install surfaces``

    Examples
    --------
    >>> from hyperactive.experiment.surfaces import Rastrigin
    >>> func = Rastrigin(n_dim=5)
    >>> score, metadata = func.score({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0})
    """

    _tags = {
        "object_type": "experiment",
        "property:randomness": "deterministic",
        "property:higher_or_lower_is_better": "lower",
        "property:function_family": "surfaces",
    }

    # Subclasses must set this to the Surfaces class to wrap
    _surfaces_class = None

    def __init__(self, **kwargs):
        # Pop metric before passing to Surfaces, store for our use
        self._metric = kwargs.get("metric", "loss")

        # Instantiate the wrapped Surfaces function
        self._surfaces_func = self._surfaces_class(**kwargs)

        super().__init__()

    def _paramnames(self):
        """Return parameter names from the wrapped Surfaces function.

        Returns
        -------
        list of str
            The parameter names expected by the wrapped function.
        """
        return sorted(self._surfaces_func.default_search_space.keys())

    def _evaluate(self, params):
        """Evaluate the parameters using the wrapped Surfaces function.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The loss value from the Surfaces function.
        dict
            Empty metadata dict (Surfaces functions don't return metadata).
        """
        # Use loss() to get a consistent minimization value
        loss = self._surfaces_func.loss(params)
        return loss, {}

    def get_default_search_space(self):
        """Return the default search space from the wrapped Surfaces function.

        Returns
        -------
        dict
            Dictionary mapping parameter names to arrays/lists of values.
        """
        return self._surfaces_func.default_search_space

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        # Default implementation - subclasses should override
        return [{}]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        # Default implementation - subclasses should override
        return [{}]
