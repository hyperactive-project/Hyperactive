"""Gramacy and Lee 1D test function wrapper."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from surfaces.test_functions.mathematical import GramacyAndLeeFunction

from hyperactive.experiment.surfaces.mathematical._base import BaseMathematicalExperiment


class GramacyLee(BaseMathematicalExperiment):
    r"""Gramacy and Lee one-dimensional test function.

    A simple one-dimensional test function commonly used for testing
    optimization algorithms and surrogate modeling techniques.

    The function is defined as:

    .. math::
        f(x) = \frac{\sin(10\pi x)}{2x} + (x - 1)^4

    The function has a global minimum at approximately x = 0.548563.

    Parameters
    ----------
    metric : str, default="loss"
        Metric mode: "loss" (minimize) or "score" (maximize).

    Attributes
    ----------
    default_bounds : tuple
        Default parameter bounds (0.5, 2.5).

    See Also
    --------
    surfaces.test_functions.mathematical.GramacyAndLeeFunction :
        The underlying Surfaces implementation.

    References
    ----------
    .. [1] Gramacy, R. B., & Lee, H. K. (2012). "Cases for the nugget in
       modeling computer experiments". Statistics and Computing, 22(3),
       713-722.

    Examples
    --------
    Basic evaluation:

    >>> from hyperactive.experiment.surfaces import GramacyLee
    >>> func = GramacyLee()
    >>> params = {"x0": 1.0}
    >>> loss, metadata = func.evaluate(params)

    Optimization with Hyperactive:

    >>> import numpy as np
    >>> from hyperactive.experiment.surfaces import GramacyLee
    >>> from hyperactive.opt.gfo import BayesianOptimizer
    >>>
    >>> func = GramacyLee()
    >>> search_space = {"x0": np.linspace(0.5, 2.5, 100)}
    >>>
    >>> optimizer = BayesianOptimizer(
    ...     search_space=search_space,
    ...     n_iter=30,
    ...     experiment=func,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "object_type": "experiment",
        "property:randomness": "deterministic",
        "property:higher_or_lower_is_better": "lower",
        "property:function_family": "surfaces",
        "property:domain": "mathematical",
        "property:dimensionality": "1d",
        "property:modality": "multimodal",
        "property:convexity": "non-convex",
        "property:differentiability": "differentiable",
    }

    _surfaces_class = GramacyAndLeeFunction

    def __init__(self, metric="loss"):
        self.metric = metric
        super().__init__(metric=metric)

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
        return [{"metric": "loss"}, {"metric": "score"}]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        return [{"x0": 1.0}, {"x0": 0.5}]
