"""GFO adapter for search space conversion.

This module provides the adapter that converts SearchSpace objects
to the format expected by Gradient-Free-Optimizers (GFO).
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any

import numpy as np

from .._dimension import DimensionType
from .._search_space import SearchSpace
from ._base import BaseSearchSpaceAdapter

__all__ = ["GFOSearchSpaceAdapter"]


class GFOSearchSpaceAdapter(BaseSearchSpaceAdapter):
    """Adapter for Gradient-Free-Optimizers (GFO).

    GFO expects search spaces as dictionaries of numpy arrays.
    Continuous dimensions are discretized into arrays.

    Parameters
    ----------
    search_space : SearchSpace
        The search space to adapt.
    resolution : int, default=100
        Number of points to use when discretizing continuous dimensions.

    Examples
    --------
    >>> from hyperactive.search_space import SearchSpace
    >>> from hyperactive.search_space.adapters import GFOSearchSpaceAdapter
    >>> space = SearchSpace(x=(0.0, 10.0), y=["a", "b", "c"])
    >>> adapter = GFOSearchSpaceAdapter(space)
    >>> gfo_space = adapter.adapt(resolution=50)
    >>> len(gfo_space["x"])  # 50 points
    50
    """

    def __init__(self, search_space: SearchSpace, resolution: int = 100):
        """Initialize the GFO adapter.

        Parameters
        ----------
        search_space : SearchSpace
            The search space to adapt.
        resolution : int, default=100
            Number of points for discretizing continuous dimensions.
        """
        super().__init__(search_space)
        self.resolution = resolution

    def adapt(self, resolution: int | None = None, **kwargs) -> dict[str, np.ndarray]:
        """Convert search space to GFO format.

        Parameters
        ----------
        resolution : int, optional
            Override default resolution for continuous discretization.
        **kwargs
            Additional options (ignored).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to numpy arrays.
        """
        if resolution is not None:
            self.resolution = resolution

        result = {}

        for name, dim in self.space.dimensions.items():
            result[name] = self._convert_dimension(dim)

        return result

    def _convert_dimension(self, dim) -> np.ndarray:
        """Convert a single dimension to numpy array.

        Parameters
        ----------
        dim : Dimension
            The dimension to convert.

        Returns
        -------
        np.ndarray
            The numpy array representation.
        """
        if dim.dim_type == DimensionType.CATEGORICAL:
            # Store as object array to preserve types (classes, functions, etc.)
            return np.array(dim.values, dtype=object)

        elif dim.dim_type == DimensionType.DISCRETE:
            # Already a numpy array
            return np.asarray(dim.values)

        elif dim.dim_type == DimensionType.CONTINUOUS:
            # Discretize continuous range
            if dim.dtype == int:
                return np.arange(dim.low, dim.high + 1, dtype=int)
            else:
                return np.linspace(dim.low, dim.high, self.resolution)

        elif dim.dim_type == DimensionType.CONTINUOUS_LOG:
            # Log-spaced discretization
            return np.logspace(
                np.log10(dim.low), np.log10(dim.high), self.resolution
            )

        elif dim.dim_type == DimensionType.DISTRIBUTION:
            # Sample from scipy distribution
            samples = dim.values.rvs(size=self.resolution)
            return np.unique(np.sort(samples))

        elif dim.dim_type == DimensionType.CONSTANT:
            # Single-element array
            return np.array([dim.values])

        else:
            raise ValueError(f"Unknown dimension type: {dim.dim_type}")

    def get_constraints(self) -> list:
        """Get constraints in GFO format (list of callables).

        GFO accepts constraints as a list of callable predicates.
        Each constraint returns True if the parameter combination is valid.

        Returns
        -------
        list[callable]
            List of constraint functions.
        """
        constraints = []

        # Add explicit constraints
        for constraint in self.space.constraints:
            constraints.append(constraint.predicate)

        # Add conditions as implicit constraints
        # (inactive params should be skipped by the optimizer)
        for condition in self.space.conditions:
            # Create a constraint that allows either:
            # 1. The condition is met (param is active)
            # 2. The param is not in the params dict

            def condition_constraint(
                params, cond=condition
            ):
                # If we can't evaluate, allow it
                if not cond.can_evaluate(params):
                    return True
                # If param is active, allow it
                return cond.is_active(params)

            constraints.append(condition_constraint)

        return constraints if constraints else None
