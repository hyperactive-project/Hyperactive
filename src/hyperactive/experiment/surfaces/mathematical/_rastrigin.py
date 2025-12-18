"""Rastrigin N-dimensional test function wrapper."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from surfaces.test_functions.mathematical import RastriginFunction

from hyperactive.experiment.surfaces.mathematical._base import BaseMathematicalExperiment


class Rastrigin(BaseMathematicalExperiment):
    r"""Rastrigin N-dimensional test function.

    A highly multimodal function with many local minima arranged in a
    regular lattice pattern. It is commonly used to test the ability
    of optimization algorithms to escape local optima.

    The function is defined as:

    .. math::
        f(\vec{x}) = An + \sum_{i=1}^{n} [x_i^2 - A\cos(\omega x_i)]

    where :math:`A = 10` and :math:`\omega = 2\pi` by default.

    The global minimum is :math:`f(\vec{0}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    A : float, default=10
        Amplitude of the cosine modulation.
    angle : float, default=2*pi
        Angular frequency parameter.
    metric : str, default="loss"
        Metric mode: "loss" (minimize) or "score" (maximize).

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    default_bounds : tuple
        Default parameter bounds (-5.0, 5.0).

    See Also
    --------
    surfaces.test_functions.mathematical.RastriginFunction :
        The underlying Surfaces implementation.

    Examples
    --------
    Basic evaluation:

    >>> from hyperactive.experiment.surfaces import Rastrigin
    >>> func = Rastrigin(n_dim=3)
    >>> params = {"x0": 0.0, "x1": 0.0, "x2": 0.0}  # Global optimum
    >>> loss, metadata = func.evaluate(params)
    >>> abs(loss) < 1e-10
    True

    Higher dimensions:

    >>> func = Rastrigin(n_dim=10)
    >>> params = {f"x{i}": 0.0 for i in range(10)}
    >>> loss, _ = func.evaluate(params)

    Optimization with Hyperactive:

    >>> import numpy as np
    >>> from hyperactive.experiment.surfaces import Rastrigin
    >>> from hyperactive.opt.gfo import ParticleSwarmOptimizer
    >>>
    >>> func = Rastrigin(n_dim=5)
    >>> search_space = {f"x{i}": np.linspace(-5, 5, 100) for i in range(5)}
    >>>
    >>> optimizer = ParticleSwarmOptimizer(
    ...     search_space=search_space,
    ...     n_iter=100,
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
        "property:dimensionality": "nd",
        "property:modality": "multimodal",
        "property:convexity": "non-convex",
        "property:differentiability": "differentiable",
        "property:separability": "separable",
        "property:global_optimum_known": True,
    }

    _surfaces_class = RastriginFunction

    def __init__(self, n_dim, A=10, angle=2 * np.pi, metric="loss"):
        self.n_dim = n_dim
        self.A = A
        self.angle = angle
        self.metric = metric
        super().__init__(n_dim=n_dim, A=A, angle=angle, metric=metric)

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
        return [
            {"n_dim": 2},
            {"n_dim": 5, "A": 5},
            {"n_dim": 10, "A": 10, "angle": np.pi, "metric": "score"},
        ]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        return [
            {"x0": 0.0, "x1": 0.0},
            {f"x{i}": 0.0 for i in range(5)},
            {f"x{i}": i * 0.1 for i in range(10)},
        ]
