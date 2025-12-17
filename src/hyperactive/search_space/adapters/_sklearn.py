"""sklearn adapter for search space conversion.

This module provides the adapter that converts SearchSpace objects
to the format expected by sklearn's RandomizedSearchCV and GridSearchCV.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any

import numpy as np

from .._dimension import DimensionType
from .._search_space import SearchSpace
from ._base import BaseSearchSpaceAdapter

__all__ = ["SklearnSearchSpaceAdapter"]


class SklearnSearchSpaceAdapter(BaseSearchSpaceAdapter):
    """Adapter for sklearn search utilities.

    sklearn uses dictionaries with lists (for grid search) or
    scipy distributions (for randomized search).

    Parameters
    ----------
    search_space : SearchSpace
        The search space to adapt.

    Examples
    --------
    >>> from hyperactive.search_space import SearchSpace
    >>> from hyperactive.search_space.adapters import SklearnSearchSpaceAdapter
    >>> space = SearchSpace(x=(0.0, 10.0), y=["a", "b", "c"])
    >>> adapter = SklearnSearchSpaceAdapter(space)
    >>> sklearn_space = adapter.adapt(mode="random")
    """

    def adapt(self, mode: str = "random", grid_resolution: int = 10, **kwargs) -> dict:
        """Convert search space to sklearn format.

        Parameters
        ----------
        mode : str, default="random"
            Either "random" for RandomizedSearchCV or "grid" for GridSearchCV.
        grid_resolution : int, default=10
            Number of points for continuous dimensions in grid mode.
        **kwargs
            Additional options (ignored).

        Returns
        -------
        dict
            Dictionary with parameter specifications for sklearn.
        """
        if mode == "random":
            return self._to_param_distributions()
        elif mode == "grid":
            return self._to_param_grid(grid_resolution)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'random' or 'grid'.")

    def _to_param_distributions(self) -> dict:
        """Convert to RandomizedSearchCV param_distributions format.

        Returns
        -------
        dict
            Dictionary mapping parameter names to distributions or lists.
        """
        import scipy.stats

        result = {}

        for name, dim in self.space.dimensions.items():
            if dim.dim_type == DimensionType.CATEGORICAL:
                result[name] = dim.values

            elif dim.dim_type == DimensionType.DISCRETE:
                result[name] = dim.values.tolist()

            elif dim.dim_type == DimensionType.CONTINUOUS:
                if dim.dtype == int:
                    result[name] = scipy.stats.randint(int(dim.low), int(dim.high) + 1)
                else:
                    result[name] = scipy.stats.uniform(dim.low, dim.high - dim.low)

            elif dim.dim_type == DimensionType.CONTINUOUS_LOG:
                result[name] = scipy.stats.loguniform(dim.low, dim.high)

            elif dim.dim_type == DimensionType.DISTRIBUTION:
                # Pass scipy distribution directly
                result[name] = dim.values

            elif dim.dim_type == DimensionType.CONSTANT:
                result[name] = [dim.values]

            else:
                raise ValueError(f"Unknown dimension type: {dim.dim_type}")

        return result

    def _to_param_grid(self, resolution: int = 10) -> dict:
        """Convert to GridSearchCV param_grid format.

        Parameters
        ----------
        resolution : int, default=10
            Number of points for continuous dimensions.

        Returns
        -------
        dict
            Dictionary mapping parameter names to lists of values.
        """
        result = {}

        for name, dim in self.space.dimensions.items():
            if dim.dim_type == DimensionType.CATEGORICAL:
                result[name] = list(dim.values)

            elif dim.dim_type == DimensionType.DISCRETE:
                result[name] = dim.values.tolist()

            elif dim.dim_type == DimensionType.CONTINUOUS:
                if dim.dtype == int:
                    result[name] = list(range(int(dim.low), int(dim.high) + 1))
                else:
                    result[name] = np.linspace(dim.low, dim.high, resolution).tolist()

            elif dim.dim_type == DimensionType.CONTINUOUS_LOG:
                result[name] = np.logspace(
                    np.log10(dim.low), np.log10(dim.high), resolution
                ).tolist()

            elif dim.dim_type == DimensionType.DISTRIBUTION:
                # Sample for grid search
                samples = dim.values.rvs(size=resolution)
                result[name] = list(set(samples.tolist()))

            elif dim.dim_type == DimensionType.CONSTANT:
                result[name] = [dim.values]

            else:
                raise ValueError(f"Unknown dimension type: {dim.dim_type}")

        return result

    def get_constraints(self) -> list:
        """Get constraints as a list of callables.

        sklearn doesn't natively support constraints, but these can be
        used for filtering parameter combinations.

        Returns
        -------
        list[callable]
            List of constraint predicates.
        """
        return [c.predicate for c in self.space.constraints]

    def filter_invalid_combinations(self, param_list: list[dict]) -> list[dict]:
        """Filter parameter combinations that violate constraints.

        Parameters
        ----------
        param_list : list[dict]
            List of parameter combinations to filter.

        Returns
        -------
        list[dict]
            List of valid parameter combinations.
        """
        valid = []
        for params in param_list:
            # Check constraints
            if not self.space.check_constraints(params):
                continue

            # Check conditions (filter inactive params)
            active_params = self.space.filter_active_params(params)

            valid.append(active_params)

        return valid
