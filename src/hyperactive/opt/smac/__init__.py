"""SMAC3 optimization algorithms.

This module provides wrappers for SMAC3 (Sequential Model-based Algorithm
Configuration), a versatile Bayesian optimization package developed by the
AutoML groups at the Universities of Hannover and Freiburg.

Available Optimizers
--------------------
SmacRandomForest
    Optimizer using Random Forest surrogate model.
    Best for mixed continuous/categorical/integer parameter spaces.

SmacGaussianProcess
    Optimizer using Gaussian Process surrogate model.
    Best for continuous parameter spaces with small to moderate budgets.

SmacRandomSearch
    Random search baseline without surrogate model.
    Useful for comparison and high-dimensional problems.

Installation
------------
SMAC3 requires additional dependencies. Install with::

    pip install smac

Or install hyperactive with SMAC support::

    pip install hyperactive[smac]

Examples
--------
>>> from hyperactive.opt.smac import SmacRandomForest
>>> from hyperactive.experiment.bench import Ackley

>>> ackley = Ackley.create_test_instance()
>>> optimizer = SmacRandomForest(
...     param_space={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
...     n_iter=50,
...     experiment=ackley,
... )
>>> best_params = optimizer.solve()  # doctest: +SKIP

References
----------
.. [1] Lindauer, M., et al. (2022). SMAC3: A Versatile Bayesian Optimization
       Package for Hyperparameter Optimization. JMLR.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._gaussian_process import SmacGaussianProcess
from ._random_forest import SmacRandomForest
from ._random_search import SmacRandomSearch

__all__ = [
    "SmacRandomForest",
    "SmacGaussianProcess",
    "SmacRandomSearch",
]
