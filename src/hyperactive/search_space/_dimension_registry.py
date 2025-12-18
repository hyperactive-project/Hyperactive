"""Dimension registry for search space dimensions.

This module provides the DimensionRegistry class which manages storage
and lookup of search space dimensions.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any, Iterator

from ._dimension import Dimension, DimensionType, infer_dimension

__all__ = ["DimensionRegistry"]


class DimensionRegistry:
    """Manages dimension storage and lookup.

    Provides a focused interface for adding, accessing, and querying
    dimensions in a search space.

    Examples
    --------
    >>> import numpy as np
    >>> registry = DimensionRegistry()
    >>> registry.add("x", np.arange(10))
    >>> registry.add("kernel", ["rbf", "linear"])
    >>> "x" in registry
    True
    >>> registry.get_dimension_types()
    {'x': DimensionType.DISCRETE, 'kernel': DimensionType.CATEGORICAL}
    """

    def __init__(self):
        """Initialize empty dimension registry."""
        self._dimensions: dict[str, Dimension] = {}

    def add(self, name: str, value: Any) -> None:
        """Add a dimension with automatic type inference.

        Parameters
        ----------
        name : str
            The parameter name.
        value : Any
            The value specification (list, tuple, array, etc.).
        """
        dimension = infer_dimension(name, value)
        self._dimensions[name] = dimension

    def add_dimension(self, dimension: Dimension) -> None:
        """Add a pre-constructed Dimension object.

        Parameters
        ----------
        dimension : Dimension
            The dimension to add.
        """
        self._dimensions[dimension.name] = dimension

    @property
    def dimensions(self) -> dict[str, Dimension]:
        """Get the dimensions dictionary.

        Returns the actual internal dict for backward compatibility.
        """
        return self._dimensions

    def __contains__(self, name: str) -> bool:
        """Check if parameter name exists."""
        return name in self._dimensions

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._dimensions)

    def __len__(self) -> int:
        """Return number of dimensions."""
        return len(self._dimensions)

    def __getitem__(self, name: str) -> Dimension:
        """Get dimension by name."""
        return self._dimensions[name]

    def get(self, name: str, default: Dimension | None = None) -> Dimension | None:
        """Get dimension by name with default.

        Parameters
        ----------
        name : str
            The parameter name.
        default : Dimension, optional
            Default value if not found.

        Returns
        -------
        Dimension or None
            The dimension or default.
        """
        return self._dimensions.get(name, default)

    def items(self):
        """Return dimension items."""
        return self._dimensions.items()

    def keys(self):
        """Return dimension names."""
        return self._dimensions.keys()

    def values(self):
        """Return dimension objects."""
        return self._dimensions.values()

    def param_names(self) -> list[str]:
        """Get list of all parameter names.

        Returns
        -------
        list[str]
            List of parameter names.
        """
        return list(self._dimensions.keys())

    def get_dimension_types(self) -> dict[str, DimensionType]:
        """Get a mapping of parameter names to their dimension types.

        Returns
        -------
        dict[str, DimensionType]
            Dictionary mapping parameter names to dimension types.
        """
        return {name: dim.dim_type for name, dim in self._dimensions.items()}

    def has_dimension_type(self, dim_type: DimensionType) -> bool:
        """Check if the registry has any dimension of the given type.

        Parameters
        ----------
        dim_type : DimensionType
            The dimension type to check for.

        Returns
        -------
        bool
            True if any dimension has the given type.
        """
        return any(dim.dim_type == dim_type for dim in self._dimensions.values())

    def copy(self) -> "DimensionRegistry":
        """Create a copy of this registry.

        Returns
        -------
        DimensionRegistry
            New registry with copies of all dimensions.
        """
        new_registry = DimensionRegistry()
        for name, dim in self._dimensions.items():
            new_registry._dimensions[name] = dim
        return new_registry
