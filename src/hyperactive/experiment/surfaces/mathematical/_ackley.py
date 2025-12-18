"""Ackley 2D test function wrapper."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from surfaces.test_functions.mathematical import AckleyFunction

from hyperactive.experiment.surfaces.mathematical._base import BaseMathematicalExperiment


class Ackley2D(BaseMathematicalExperiment):
    r"""Ackley two-dimensional test function.

    A non-convex function used as a performance test problem for optimization
    algorithms. It has a nearly flat outer region with a large hole at the
    center, making it challenging for optimization methods.

    The function is defined as:

    .. math::
        f(x, y) = -A \exp\left[-0.2\sqrt{0.5(x^2+y^2)}\right]
        - \exp\left[0.5(\cos \omega x + \cos \omega y)\right] + e + A

    where :math:`A = 20` and :math:`\omega = 2\pi` by default.

    The global minimum is :math:`f(0, 0) = 0`.

    Parameters
    ----------
    A : float, default=20
        Amplitude parameter.
    angle : float, default=2*pi
        Angular frequency parameter.
    metric : str, default="loss"
        Metric mode: "loss" (minimize) or "score" (maximize).

    Attributes
    ----------
    default_bounds : tuple
        Default parameter bounds (-5.0, 5.0).

    See Also
    --------
    surfaces.test_functions.mathematical.AckleyFunction :
        The underlying Surfaces implementation.
    hyperactive.experiment.bench.Ackley :
        Hyperactive's native N-dimensional Ackley implementation.

    References
    ----------
    .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic
       hillclimbing". Kluwer Academic Publishers, Boston MA.

    Examples
    --------
    Basic evaluation:

    >>> from hyperactive.experiment.surfaces import Ackley2D
    >>> func = Ackley2D()
    >>> params = {"x0": 0.0, "x1": 0.0}  # Global optimum
    >>> loss, metadata = func.evaluate(params)
    >>> abs(loss) < 1e-10
    True

    Optimization with Hyperactive:

    >>> import numpy as np
    >>> from hyperactive.experiment.surfaces import Ackley2D
    >>> from hyperactive.opt.gfo import HillClimbing
    >>>
    >>> func = Ackley2D()
    >>> search_space = {
    ...     "x0": np.linspace(-5, 5, 100),
    ...     "x1": np.linspace(-5, 5, 100),
    ... }
    >>>
    >>> optimizer = HillClimbing(
    ...     search_space=search_space,
    ...     n_iter=50,
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
        "property:dimensionality": "2d",
        "property:modality": "multimodal",
        "property:convexity": "non-convex",
        "property:differentiability": "differentiable",
        "property:separability": "non-separable",
        "property:global_optimum_known": True,
    }

    _surfaces_class = AckleyFunction

    def __init__(self, A=20, angle=2 * np.pi, metric="loss"):
        self.A = A
        self.angle = angle
        self.metric = metric
        super().__init__(A=A, angle=angle, metric=metric)

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
            {"metric": "loss"},
            {"A": 10, "metric": "score"},
            {"A": 20, "angle": np.pi, "metric": "loss"},
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
            {"x0": 1.0, "x1": 1.0},
            {"x0": -2.5, "x1": 2.5},
        ]
