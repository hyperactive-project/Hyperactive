"""Dimension classes for search space specification.

This module provides the internal representation of search space dimensions
and automatic type inference from Python values.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


class DimensionType(Enum):
    """Enumeration of dimension types in a search space."""

    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    CONTINUOUS_LOG = "continuous_log"
    DISTRIBUTION = "distribution"
    CONSTANT = "constant"


@dataclass
class Dimension:
    """Internal representation of a search space dimension.

    Parameters
    ----------
    name : str
        The name of the dimension (parameter name).
    values : Any
        The raw values, bounds, or distribution object.
    dim_type : DimensionType
        The type of dimension.
    dtype : type
        The Python type of values in this dimension (int, float, str, etc.).
    low : float or None, default=None
        Lower bound for continuous dimensions.
    high : float or None, default=None
        Upper bound for continuous dimensions.
    log_scale : bool, default=False
        Whether this dimension uses log scale.
    """

    name: str
    values: Any
    dim_type: DimensionType
    dtype: type
    low: float | None = None
    high: float | None = None
    log_scale: bool = False

    def __repr__(self) -> str:
        """Return string representation."""
        type_str = self.dim_type.value
        if self.log_scale:
            type_str += "(log)"
        return f"Dimension({self.name!r}, {type_str})"


def infer_dimension(name: str, value: Any) -> Dimension:
    """Infer dimension type from Python value.

    This function implements the type inference rules for SearchSpace:
    - list -> categorical
    - tuple(low, high) with floats -> continuous float
    - tuple(low, high) with ints -> continuous int
    - tuple(low, high, "log") -> continuous log-scale
    - np.ndarray -> discrete
    - scipy.stats rv_frozen -> distribution
    - optuna distribution -> distribution
    - scalar -> constant

    Parameters
    ----------
    name : str
        The parameter name.
    value : Any
        The value specification for this parameter.

    Returns
    -------
    Dimension
        The inferred dimension object.

    Raises
    ------
    TypeError
        If the value type is not supported.
    ValueError
        If the tuple specification is invalid.
    """
    # Check for scipy distribution first (has .rvs method)
    if _is_scipy_distribution(value):
        return Dimension(
            name=name,
            values=value,
            dim_type=DimensionType.DISTRIBUTION,
            dtype=float,
        )

    # Check for optuna distribution
    if _is_optuna_distribution(value):
        return Dimension(
            name=name,
            values=value,
            dim_type=DimensionType.DISTRIBUTION,
            dtype=float,
        )

    # List -> categorical
    if isinstance(value, list):
        dtype = _infer_list_dtype(value)
        return Dimension(
            name=name,
            values=value,
            dim_type=DimensionType.CATEGORICAL,
            dtype=dtype,
        )

    # Tuple -> continuous
    if isinstance(value, tuple):
        return _infer_tuple_dimension(name, value)

    # Numpy array -> discrete
    if isinstance(value, np.ndarray):
        # Determine dtype from array
        if np.issubdtype(value.dtype, np.integer):
            dtype = int
        elif np.issubdtype(value.dtype, np.floating):
            dtype = float
        else:
            dtype = object
        return Dimension(
            name=name,
            values=value,
            dim_type=DimensionType.DISCRETE,
            dtype=dtype,
            low=float(value.min()) if value.size > 0 else None,
            high=float(value.max()) if value.size > 0 else None,
        )

    # Scalar -> constant
    if isinstance(value, (int, float, str, bool, type(None))):
        return Dimension(
            name=name,
            values=value,
            dim_type=DimensionType.CONSTANT,
            dtype=type(value) if value is not None else type(None),
        )

    # Unknown type
    raise TypeError(
        f"Cannot infer dimension type for parameter '{name}' with value {value!r} "
        f"of type {type(value).__name__}. "
        f"Expected: list, tuple, numpy.ndarray, scipy distribution, or scalar."
    )


def _infer_tuple_dimension(name: str, value: tuple) -> Dimension:
    """Infer dimension from tuple specification.

    Parameters
    ----------
    name : str
        The parameter name.
    value : tuple
        The tuple specification: (low, high) or (low, high, "log").

    Returns
    -------
    Dimension
        The inferred dimension object.

    Raises
    ------
    ValueError
        If the tuple format is invalid.
    """
    if len(value) == 2:
        low, high = value

        # Both ints -> continuous integer
        if isinstance(low, int) and isinstance(high, int):
            return Dimension(
                name=name,
                values=value,
                dim_type=DimensionType.CONTINUOUS,
                dtype=int,
                low=low,
                high=high,
            )

        # At least one float -> continuous float
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return Dimension(
                name=name,
                values=value,
                dim_type=DimensionType.CONTINUOUS,
                dtype=float,
                low=float(low),
                high=float(high),
            )

    elif len(value) == 3:
        low, high, scale = value

        if scale == "log":
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise ValueError(
                    f"Log-scale bounds for parameter '{name}' must be numeric, "
                    f"got {type(low).__name__} and {type(high).__name__}."
                )
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"Log-scale bounds for parameter '{name}' must be positive, "
                    f"got low={low}, high={high}."
                )
            return Dimension(
                name=name,
                values=(low, high),
                dim_type=DimensionType.CONTINUOUS_LOG,
                dtype=float,
                low=float(low),
                high=float(high),
                log_scale=True,
            )

    raise ValueError(
        f"Invalid tuple specification for parameter '{name}': {value}. "
        f"Expected (low, high) or (low, high, 'log')."
    )


def _is_scipy_distribution(value: Any) -> bool:
    """Check if value is a scipy.stats frozen distribution.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if value is a scipy frozen distribution.
    """
    # scipy frozen distributions have .rvs(), .pdf()/.pmf(), .cdf() methods
    return (
        callable(getattr(value, "rvs", None))
        and hasattr(value, "args")
        and hasattr(value, "kwds")
    )


def _is_optuna_distribution(value: Any) -> bool:
    """Check if value is an Optuna distribution.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if value is an Optuna distribution.
    """
    try:
        import optuna.distributions

        return isinstance(value, optuna.distributions.BaseDistribution)
    except ImportError:
        return False


def _infer_list_dtype(values: list) -> type:
    """Infer the dtype of a categorical list.

    Parameters
    ----------
    values : list
        The list of categorical values.

    Returns
    -------
    type
        The inferred dtype (str, int, float, or object for mixed/complex types).
    """
    if not values:
        return object

    # Filter out None values for type inference
    non_none_values = [v for v in values if v is not None]
    if not non_none_values:
        return type(None)

    types = set(type(v) for v in non_none_values)

    # Single type
    if len(types) == 1:
        return types.pop()

    # Mixed numeric types -> float
    if types <= {int, float}:
        return float

    # Mixed types or complex types (classes, functions) -> object
    return object


def make_prefix(value: Any) -> str:
    """Create a safe prefix string from a value (e.g., class name).

    Used for creating prefixed parameter names in nested search spaces.
    For example, RandomForestClassifier becomes "randomforestclassifier".

    Parameters
    ----------
    value : Any
        The value to create a prefix from. Typically a class or callable.

    Returns
    -------
    str
        A lowercase, underscore-separated prefix string.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> make_prefix(RandomForestClassifier)
    'randomforestclassifier'
    >>> make_prefix(lambda x: x)
    '<lambda>'
    >>> make_prefix("My Model")
    'my_model'
    """
    if isinstance(value, type):
        return value.__name__.lower()
    elif callable(value):
        return getattr(value, "__name__", str(value)).lower()
    return str(value).lower().replace(" ", "_").replace("-", "_")
