"""Base class for Surfaces mathematical test function wrappers."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.surfaces._base import BaseSurfacesExperiment


class BaseMathematicalExperiment(BaseSurfacesExperiment):
    """Base class for wrapping Surfaces mathematical test functions.

    Mathematical test functions compute a loss value based on input parameters.
    They are used as benchmarks for optimization algorithms. The global optimum
    is typically known and used to assess algorithm performance.

    Subclasses must set:
        _surfaces_class : class
            The Surfaces mathematical function class to wrap.

    Parameters
    ----------
    metric : str, default="loss"
        Metric mode: "loss" (minimize) or "score" (maximize).
        Controls the return value of the wrapped function's ``__call__``.

    Notes
    -----
    Mathematical functions naturally return loss values (lower is better),
    so ``property:higher_or_lower_is_better`` is set to "lower".

    Examples
    --------
    >>> from hyperactive.experiment.surfaces import Rastrigin
    >>> func = Rastrigin(n_dim=3)
    >>> params = {"x0": 0, "x1": 0, "x2": 0}  # Global optimum
    >>> loss, _ = func.evaluate(params)
    >>> abs(loss) < 1e-10
    True
    """

    _tags = {
        "object_type": "experiment",
        "property:randomness": "deterministic",
        "property:higher_or_lower_is_better": "lower",
        "property:function_family": "surfaces",
        "property:domain": "mathematical",
    }

    def __init__(self, metric="loss", **kwargs):
        super().__init__(metric=metric, **kwargs)
