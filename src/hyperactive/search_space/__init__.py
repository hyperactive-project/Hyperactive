"""Unified search space specification for Hyperactive.

This module provides a unified API for specifying search spaces that
works across all Hyperactive optimizer backends (GFO, Optuna, sklearn).

Main Classes
------------
SearchSpace
    The main class for defining search spaces with type inference.
Dimension
    Internal representation of a search space dimension.
DimensionType
    Enumeration of dimension types.
Condition
    Specifies when a dimension is active (conditional search spaces).
Constraint
    Defines constraints on valid parameter combinations.

Examples
--------
Basic usage with type inference:

>>> import numpy as np
>>> from hyperactive.search_space import SearchSpace
>>> space = SearchSpace(
...     x=np.arange(-10, 10, 0.1),       # discrete
...     lr=(1e-5, 1e-1, "log"),          # log-scale continuous
...     kernel=["rbf", "linear", "poly"], # categorical
...     seed=42,                          # constant
... )

With conditions:

>>> space = SearchSpace(
...     kernel=["rbf", "linear", "poly"],
...     gamma=(1e-4, 10.0, "log"),
... )
>>> space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")

With constraints:

>>> space.add_constraint(lambda p: p["x"] + p["y"] < 10)

Union of search spaces:

>>> combined = space1 | space2
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._condition import Condition
from ._constraint import Constraint
from ._dimension import Dimension, DimensionType, infer_dimension
from ._search_space import SearchSpace

__all__ = [
    "SearchSpace",
    "Dimension",
    "DimensionType",
    "Condition",
    "Constraint",
    "infer_dimension",
]
