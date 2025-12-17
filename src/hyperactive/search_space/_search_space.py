"""SearchSpace class for unified search space specification.

This module provides the main SearchSpace class that enables a unified
API for specifying search spaces across all Hyperactive optimizers.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._condition import Condition
from ._constraint import Constraint
from ._dimension import Dimension, DimensionType, infer_dimension

__all__ = ["SearchSpace"]


class SearchSpace:
    """Unified search space for Hyperactive optimizers.

    SearchSpace provides a clean, Pythonic API for specifying parameter
    search spaces with automatic type inference. It works across all
    Hyperactive backends (GFO, Optuna, sklearn) via adaptation layers.

    Parameters can be specified as keyword arguments with automatic type
    inference, or as a dictionary for backward compatibility.

    Type Inference Rules
    --------------------
    - list -> Categorical dimension
    - tuple(low, high) with floats -> Continuous float dimension
    - tuple(low, high) with ints -> Continuous integer dimension
    - tuple(low, high, "log") -> Log-scale continuous dimension
    - numpy.ndarray -> Discrete dimension
    - scipy.stats distribution -> Distribution dimension
    - scalar (int, float, str) -> Constant (not searched)

    Parameters
    ----------
    __dict_space : dict, optional
        Dictionary-based space specification (for backward compatibility).
        If provided, keys are parameter names and values follow the type
        inference rules above.
    **kwargs
        Keyword arguments defining dimensions. Each keyword argument
        follows the type inference rules.

    Attributes
    ----------
    dimensions : dict[str, Dimension]
        Dictionary mapping parameter names to Dimension objects.
    conditions : list[Condition]
        List of conditions that control when parameters are active.
    constraints : list[Constraint]
        List of constraints that filter invalid parameter combinations.
    nested_spaces : dict[str, dict[Any, SearchSpace]]
        Nested search spaces for hierarchical parameter structures.

    Examples
    --------
    Basic usage with type inference:

    >>> import numpy as np
    >>> space = SearchSpace(
    ...     x=np.arange(-10, 10, 0.1),       # discrete
    ...     lr=(1e-5, 1e-1, "log"),          # log-scale continuous
    ...     kernel=["rbf", "linear", "poly"], # categorical
    ...     seed=42,                          # constant
    ... )

    With conditions (method chaining):

    >>> space = SearchSpace(
    ...     kernel=["rbf", "linear", "poly"],
    ...     gamma=(1e-4, 10.0, "log"),
    ...     degree=[2, 3, 4, 5],
    ... )
    >>> space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")
    >>> space.add_condition("degree", when=lambda p: p["kernel"] == "poly")

    With constraints:

    >>> space.add_constraint(lambda p: p["x"] + p["y"] < 10)

    Union operation:

    >>> combined = space1 | space2

    Nested search spaces (estimator selection):

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> space = SearchSpace(
    ...     estimator={
    ...         RandomForestClassifier: {
    ...             "n_estimators": np.arange(10, 101, 10),
    ...             "max_depth": [3, 5, 10, None],
    ...         },
    ...         SVC: {
    ...             "C": (0.01, 100.0, "log"),
    ...             "kernel": ["rbf", "linear"],
    ...         },
    ...     },
    ... )

    The nested space is detected automatically when a dict has class keys
    mapping to parameter dicts. This creates a categorical "estimator"
    dimension plus conditional parameters for each estimator type.
    """

    def __init__(
        self,
        __dict_space: dict[str, Any] | None = None,
        /,
        **kwargs,
    ):
        """Initialize SearchSpace.

        Parameters
        ----------
        __dict_space : dict, optional
            Dictionary-based space (for backward compatibility).
        **kwargs
            Keyword arguments defining dimensions.
        """
        self.dimensions: dict[str, Dimension] = {}
        self.conditions: list[Condition] = []
        self.constraints: list[Constraint] = []
        self.nested_spaces: dict[str, dict[Any, "SearchSpace"]] = {}

        # Process dict-based space
        if __dict_space is not None:
            for name, value in __dict_space.items():
                self._add_dimension(name, value)

        # Process keyword arguments
        for name, value in kwargs.items():
            # Check if this looks like a nested space (dict with class keys)
            if isinstance(value, dict) and self._looks_like_nested_space(value):
                self._add_nested_space(name, value)
            else:
                self._add_dimension(name, value)

    def _looks_like_nested_space(self, value: dict) -> bool:
        """Check if a dict looks like a nested search space.

        A nested search space has keys that are classes or functions,
        and values that are dicts (parameter specifications).

        Parameters
        ----------
        value : dict
            The dictionary to check.

        Returns
        -------
        bool
            True if this looks like a nested space specification.
        """
        if not value:
            return False

        for key, val in value.items():
            # Keys should be classes or functions (hashable callables)
            if not (isinstance(key, type) or callable(key)):
                return False
            # Values should be dicts (parameter specifications)
            if not isinstance(val, dict):
                return False
        return True

    def _add_dimension(self, name: str, value: Any) -> None:
        """Add a dimension with automatic type inference.

        Parameters
        ----------
        name : str
            The parameter name.
        value : Any
            The value specification.
        """
        dimension = infer_dimension(name, value)
        self.dimensions[name] = dimension

    def _add_nested_space(self, name: str, nested: dict[Any, dict]) -> None:
        """Add a nested search space.

        Nested spaces are detected automatically when a dict has class/callable
        keys mapping to parameter dicts. For example:

            estimator={
                RandomForestClassifier: {"n_estimators": [10, 50, 100]},
                SVC: {"C": (0.01, 100.0, "log")},
            }

        This creates:
        - A categorical dimension "estimator" with RFC and SVC as choices
        - Flattened parameters like "randomforestclassifier__n_estimators"
        - Automatic conditions so each param is only active for its estimator

        Parameters
        ----------
        name : str
            The parameter name (e.g., "estimator").
        nested : dict
            Dictionary mapping keys (classes/functions) to parameter dicts.
        """
        # Convert nested dicts to SearchSpaces
        converted = {}
        for key, subspace_dict in nested.items():
            converted[key] = SearchSpace(subspace_dict)

        self.nested_spaces[name] = converted

        # Create parent categorical dimension from keys
        parent_values = list(nested.keys())
        self.dimensions[name] = Dimension(
            name=name,
            values=parent_values,
            dim_type=DimensionType.CATEGORICAL,
            dtype=object,
        )

        # Expand nested space to flat dimensions with conditions
        self._expand_nested_space(name, converted)

    def _expand_nested_space(
        self,
        parent_name: str,
        nested: dict[Any, "SearchSpace"],
    ) -> None:
        """Expand nested space to flat dimensions with auto-generated conditions.

        Parameters
        ----------
        parent_name : str
            The parent parameter name.
        nested : dict
            Dictionary mapping parent values to SearchSpace objects.
        """
        for parent_value, subspace in nested.items():
            # Create prefix for flattened parameter names
            prefix = self._make_prefix(parent_value)

            for dim_name, dim in subspace.dimensions.items():
                flat_name = f"{prefix}__{dim_name}"

                # Create a new dimension with the prefixed name
                new_dim = Dimension(
                    name=flat_name,
                    values=dim.values,
                    dim_type=dim.dim_type,
                    dtype=dim.dtype,
                    low=dim.low,
                    high=dim.high,
                    log_scale=dim.log_scale,
                )
                self.dimensions[flat_name] = new_dim

                # Add condition: only active when parent == this value
                # Use default argument to capture parent_value correctly in closure
                self.conditions.append(
                    Condition(
                        target_param=flat_name,
                        predicate=lambda p, pn=parent_name, pv=parent_value: p.get(pn)
                        == pv,
                        depends_on=[parent_name],
                        name=f"{flat_name}_when_{parent_name}=={self._make_prefix(parent_value)}",
                    )
                )

    def _make_prefix(self, value: Any) -> str:
        """Create a safe prefix from a value (e.g., class name).

        Parameters
        ----------
        value : Any
            The value to create a prefix from.

        Returns
        -------
        str
            A safe prefix string.
        """
        if isinstance(value, type):
            return value.__name__.lower()
        elif callable(value):
            return getattr(value, "__name__", str(value)).lower()
        return str(value).lower().replace(" ", "_").replace("-", "_")

    def add_condition(
        self,
        param: str,
        *,
        when: Callable[[dict], bool],
        depends_on: str | list[str] | None = None,
        name: str | None = None,
    ) -> "SearchSpace":
        """Add a condition for when a parameter is active.

        Parameters
        ----------
        param : str
            Name of the parameter this condition controls.
        when : callable
            Function that takes params dict and returns True if param is active.
        depends_on : str or list[str], optional
            Parameter(s) this condition depends on (for optimization).
        name : str, optional
            Optional name for the condition.

        Returns
        -------
        SearchSpace
            Self, for method chaining.

        Raises
        ------
        ValueError
            If the parameter name is unknown.

        Examples
        --------
        >>> space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")
        >>> space.add_condition(
        ...     "degree",
        ...     when=lambda p: p["kernel"] == "poly",
        ...     depends_on="kernel",
        ... )
        """
        if param not in self.dimensions:
            raise ValueError(
                f"Unknown parameter: {param}. "
                f"Available parameters: {list(self.dimensions.keys())}"
            )

        if depends_on is None:
            depends_on = []
        elif isinstance(depends_on, str):
            depends_on = [depends_on]

        self.conditions.append(
            Condition(
                target_param=param,
                predicate=when,
                depends_on=depends_on,
                name=name,
            )
        )

        return self

    def add_constraint(
        self,
        constraint: Callable[[dict], bool],
        *,
        name: str | None = None,
        params: list[str] | None = None,
    ) -> "SearchSpace":
        """Add a constraint that filters invalid parameter combinations.

        Parameters
        ----------
        constraint : callable
            Function that takes params dict and returns True if valid.
        name : str, optional
            Name for the constraint (for debugging).
        params : list[str], optional
            Parameter names involved in this constraint.

        Returns
        -------
        SearchSpace
            Self, for method chaining.

        Examples
        --------
        >>> space.add_constraint(lambda p: p["x"] + p["y"] < 10)
        >>> space.add_constraint(
        ...     lambda p: p["batch_size"] * p["lr"] < 0.5,
        ...     name="batch_lr_limit",
        ... )
        """
        if name is None:
            name = f"constraint_{len(self.constraints)}"

        self.constraints.append(
            Constraint(
                predicate=constraint,
                name=name,
                params=params,
            )
        )

        return self

    def __or__(self, other: "SearchSpace") -> "SearchSpace":
        """Union of two search spaces using | operator.

        Parameters
        ----------
        other : SearchSpace
            The search space to merge with.

        Returns
        -------
        SearchSpace
            A new search space with merged parameters.
        """
        return self.union(other)

    def union(
        self,
        other: "SearchSpace",
        on_conflict: str = "last",
    ) -> "SearchSpace":
        """Merge parameters from another search space.

        Parameters
        ----------
        other : SearchSpace
            Search space to merge.
        on_conflict : str, default="last"
            How to handle conflicting parameters:
            - "last": Use value from `other` (default)
            - "first": Keep value from `self`
            - "error": Raise ValueError

        Returns
        -------
        SearchSpace
            New search space with merged parameters.

        Raises
        ------
        ValueError
            If on_conflict="error" and there are conflicting parameters.

        Examples
        --------
        >>> base = SearchSpace(lr=(1e-5, 1e-1, "log"), batch_size=[32, 64])
        >>> reg = SearchSpace(dropout=(0.0, 0.5), weight_decay=(0.0, 0.1))
        >>> combined = base | reg
        """
        result = SearchSpace()

        # Copy dimensions from self
        for name, dim in self.dimensions.items():
            result.dimensions[name] = dim

        # Add/merge dimensions from other
        for name, dim in other.dimensions.items():
            if name in result.dimensions:
                if on_conflict == "last":
                    result.dimensions[name] = dim
                elif on_conflict == "first":
                    pass  # Keep existing
                elif on_conflict == "error":
                    raise ValueError(f"Conflicting parameter: {name}")
                else:
                    raise ValueError(
                        f"Invalid on_conflict value: {on_conflict}. "
                        f"Expected 'last', 'first', or 'error'."
                    )
            else:
                result.dimensions[name] = dim

        # Merge conditions and constraints
        result.conditions = self.conditions.copy() + other.conditions.copy()
        result.constraints = self.constraints.copy() + other.constraints.copy()

        # Merge nested spaces
        result.nested_spaces = {**self.nested_spaces, **other.nested_spaces}

        return result

    @property
    def param_names(self) -> list[str]:
        """Get list of all parameter names.

        Returns
        -------
        list[str]
            List of parameter names.
        """
        return list(self.dimensions.keys())

    @property
    def has_conditions(self) -> bool:
        """Check if the search space has any conditions.

        Returns
        -------
        bool
            True if there are conditions.
        """
        return len(self.conditions) > 0

    @property
    def has_constraints(self) -> bool:
        """Check if the search space has any constraints.

        Returns
        -------
        bool
            True if there are constraints.
        """
        return len(self.constraints) > 0

    @property
    def has_nested_spaces(self) -> bool:
        """Check if the search space has nested spaces.

        Returns
        -------
        bool
            True if there are nested spaces.
        """
        return len(self.nested_spaces) > 0

    def get_dimension_types(self) -> dict[str, DimensionType]:
        """Get a mapping of parameter names to their dimension types.

        Returns
        -------
        dict[str, DimensionType]
            Dictionary mapping parameter names to dimension types.
        """
        return {name: dim.dim_type for name, dim in self.dimensions.items()}

    def has_dimension_type(self, dim_type: DimensionType) -> bool:
        """Check if the search space has any dimension of the given type.

        Parameters
        ----------
        dim_type : DimensionType
            The dimension type to check for.

        Returns
        -------
        bool
            True if any dimension has the given type.
        """
        return any(dim.dim_type == dim_type for dim in self.dimensions.values())

    def filter_active_params(self, params: dict) -> dict:
        """Filter parameters to only include active ones based on conditions.

        Parameters
        ----------
        params : dict
            The full parameter dictionary.

        Returns
        -------
        dict
            Dictionary with only active parameters.
        """
        active_params = {}

        for name, value in params.items():
            if name not in self.dimensions:
                continue

            # Check if any condition disables this parameter
            is_active = True
            for condition in self.conditions:
                if condition.target_param == name:
                    if condition.can_evaluate(params) and not condition.is_active(
                        params
                    ):
                        is_active = False
                        break

            if is_active:
                active_params[name] = value

        return active_params

    def check_constraints(self, params: dict) -> bool:
        """Check if all constraints are satisfied.

        Parameters
        ----------
        params : dict
            The parameter values to check.

        Returns
        -------
        bool
            True if all constraints are satisfied.
        """
        return all(c.is_satisfied(params) for c in self.constraints)

    def to_backend(self, backend: str, **kwargs):
        """Convert to backend-specific format.

        Parameters
        ----------
        backend : str
            One of: "gfo", "optuna", "sklearn_grid", "sklearn_random".
        **kwargs
            Backend-specific options (e.g., resolution for GFO).

        Returns
        -------
        Any
            Adapted search space for the specified backend.

        Examples
        --------
        >>> space = SearchSpace(x=(0.0, 10.0), y=["a", "b"])
        >>> gfo_space = space.to_backend("gfo", resolution=50)
        >>> optuna_space = space.to_backend("optuna")
        """
        from .adapters import get_adapter

        adapter = get_adapter(backend, self, **kwargs)
        return adapter.adapt(**kwargs)

    def __len__(self) -> int:
        """Return the number of dimensions."""
        return len(self.dimensions)

    def __contains__(self, name: str) -> bool:
        """Check if a parameter name is in the search space."""
        return name in self.dimensions

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self.dimensions)

    def __repr__(self) -> str:
        """Return string representation."""
        dims = ", ".join(
            f"{k}={v.dim_type.value}" for k, v in self.dimensions.items()
        )
        extras = []
        if self.conditions:
            extras.append(f"{len(self.conditions)} conditions")
        if self.constraints:
            extras.append(f"{len(self.constraints)} constraints")
        if self.nested_spaces:
            extras.append(f"{len(self.nested_spaces)} nested spaces")

        extra_str = f" [{', '.join(extras)}]" if extras else ""
        return f"SearchSpace({dims}){extra_str}"
