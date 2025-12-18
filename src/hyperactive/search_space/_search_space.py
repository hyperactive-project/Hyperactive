"""SearchSpace class for unified search space specification.

This module provides the main SearchSpace class that enables a unified
API for specifying search spaces across all Hyperactive optimizers.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ._condition import Condition
from ._constraint import Constraint
from ._dimension import Dimension, DimensionType, infer_dimension, make_prefix

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
            if isinstance(value, dict):
                if not value:
                    # Empty dict is likely an error - user probably meant to define
                    # a nested space but forgot to add options
                    raise ValueError(
                        f"Empty dict provided for parameter '{name}'. "
                        f"If you intended to define a nested space, provide at least "
                        f"one option. Example: {name}={{SomeClass: {{'param': [1, 2]}}}}"
                    )
                if self._looks_like_nested_space(value):
                    self._add_nested_space(name, value)
                else:
                    self._add_dimension(name, value)
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

    def _validate_unique_prefixes(self, name: str, nested: dict[Any, dict]) -> None:
        """Validate that all nested space keys produce unique prefixes.

        Parameters
        ----------
        name : str
            The parameter name (e.g., "estimator").
        nested : dict
            Dictionary mapping keys to parameter dicts.

        Raises
        ------
        ValueError
            If multiple keys produce the same prefix.
        """
        prefix_to_keys: dict[str, list] = {}
        for key in nested.keys():
            prefix = make_prefix(key)
            if prefix not in prefix_to_keys:
                prefix_to_keys[prefix] = []
            prefix_to_keys[prefix].append(key)

        # Check for collisions
        collisions = {p: keys for p, keys in prefix_to_keys.items() if len(keys) > 1}
        if collisions:
            collision_details = []
            for prefix, keys in collisions.items():
                key_reprs = [repr(k) for k in keys]
                collision_details.append(f"  prefix '{prefix}__': {', '.join(key_reprs)}")

            raise ValueError(
                f"Nested space '{name}' has keys that produce the same prefix, "
                f"which would cause parameter name collisions:\n"
                + "\n".join(collision_details)
                + "\n\nUse named functions or classes instead of lambdas, "
                "or ensure all keys have unique names."
            )

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

        Raises
        ------
        ValueError
            If multiple keys produce the same prefix (e.g., multiple lambdas).
        """
        # Validate that all keys produce unique prefixes
        self._validate_unique_prefixes(name, nested)

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
            prefix = make_prefix(parent_value)

            for dim_name, dim in subspace.dimensions.items():
                flat_name = f"{prefix}__{dim_name}"

                # Create a new dimension with the prefixed name
                # Using dataclasses.replace() ensures all fields are copied
                new_dim = replace(dim, name=flat_name)
                self.dimensions[flat_name] = new_dim

                # Add condition: only active when parent == this value
                # Use default argument to capture parent_value correctly in closure
                self.conditions.append(
                    Condition(
                        target_param=flat_name,
                        predicate=lambda p, pn=parent_name, pv=parent_value: p.get(pn)
                        == pv,
                        depends_on=[parent_name],
                        name=f"{flat_name}_when_{parent_name}=={make_prefix(parent_value)}",
                    )
                )

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
            If the parameter name is unknown, or if adding this condition
            would create a circular dependency.

        Notes
        -----
        **Circular dependencies are not allowed.** If parameter A depends on B,
        then B cannot depend on A (directly or transitively). For example:

        - Direct cycle: A depends on B, B depends on A
        - Transitive cycle: A depends on B, B depends on C, C depends on A

        Circular dependencies would make it impossible to determine which
        parameters are active, since each would require the other to be
        evaluated first. This is detected and raises a ValueError immediately.

        Diamond-shaped dependencies ARE allowed (no cycle):

        - root -> branch1 -> leaf
        - root -> branch2 -> leaf

        **Transitive conditions:** Each condition is evaluated independently.
        If A conditions on B, and B conditions on C, the system does NOT
        automatically make A inactive when C is inactive. To achieve this,
        combine the checks in a single predicate:

            space.add_condition("A", when=lambda p: p["C"] and p["B"])

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

        # Validate that all depends_on parameters exist in the search space.
        # Without this check, typos in dependency names would silently cause
        # conditions to never evaluate (can_evaluate() returns False).
        unknown_deps = set(depends_on) - set(self.dimensions.keys())
        if unknown_deps:
            raise ValueError(
                f"Unknown parameters in depends_on: {unknown_deps}. "
                f"Available parameters: {list(self.dimensions.keys())}"
            )

        self.conditions.append(
            Condition(
                target_param=param,
                predicate=when,
                depends_on=depends_on,
                name=name,
            )
        )

        # Check for circular dependencies
        self._check_circular_dependencies()

        return self

    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in conditions.

        Builds a dependency graph from conditions and detects cycles using DFS.

        Raises
        ------
        ValueError
            If circular dependencies are detected.
        """
        # Build adjacency list: param -> list of params it depends on
        # Note: A condition on param P with depends_on=[D1, D2] means
        # P depends on D1 and D2 (edges: P->D1, P->D2)
        graph: dict[str, set[str]] = {}

        for condition in self.conditions:
            target = condition.target_param
            if target not in graph:
                graph[target] = set()
            for dep in condition.depends_on:
                graph[target].add(dep)
                # Ensure dependency nodes exist in graph
                if dep not in graph:
                    graph[dep] = set()

        # DFS-based cycle detection
        # States: 0 = unvisited, 1 = in current path, 2 = fully processed
        state: dict[str, int] = {node: 0 for node in graph}
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            """Return cycle path if found, None otherwise."""
            if state[node] == 1:
                # Found cycle - return the cycle path
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            if state[node] == 2:
                return None

            state[node] = 1
            path.append(node)

            for neighbor in graph[node]:
                if neighbor in graph:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle

            path.pop()
            state[node] = 2
            return None

        for node in graph:
            if state[node] == 0:
                cycle = dfs(node)
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    raise ValueError(
                        f"Circular dependency detected in conditions: {cycle_str}\n"
                        f"Parameter conditions cannot form cycles."
                    )

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

    def _get_nested_space_prefixes(
        self, parent_name: str, nested: dict[Any, "SearchSpace"]
    ) -> set[str]:
        """Get all prefixes used by a nested space's flattened dimensions.

        Parameters
        ----------
        parent_name : str
            The parent parameter name.
        nested : dict
            The nested space mapping parent values to SearchSpaces.

        Returns
        -------
        set[str]
            Set of prefixes (e.g., {"randomforestclassifier__", "svc__"}).
        """
        prefixes = set()
        for parent_value in nested.keys():
            prefixes.add(make_prefix(parent_value) + "__")
        return prefixes

    def union(
        self,
        other: "SearchSpace",
        on_conflict: str = "last",
        allow_type_change: bool = False,
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
        allow_type_change : bool, default=False
            If False (default), raise ValueError when a parameter's dimension
            type would change during the merge (e.g., categorical -> continuous).
            Set to True to allow type changes (rarely needed).

        Returns
        -------
        SearchSpace
            New search space with merged parameters.

        Raises
        ------
        ValueError
            If on_conflict="error" and there are conflicting parameters.
            If allow_type_change=False and a parameter's type would change.

        Examples
        --------
        >>> base = SearchSpace(lr=(1e-5, 1e-1, "log"), batch_size=[32, 64])
        >>> reg = SearchSpace(dropout=(0.0, 0.5), weight_decay=(0.0, 0.1))
        >>> combined = base | reg
        """
        result = SearchSpace()

        # Identify conflicting nested spaces and determine which prefixes to exclude.
        # When both spaces have the same nested space key (e.g., "estimator"),
        # we must handle the flattened dimensions consistently with nested_spaces.
        self_prefixes_to_exclude: set[str] = set()
        other_prefixes_to_exclude: set[str] = set()

        for parent_name in other.nested_spaces:
            if parent_name in self.nested_spaces:
                if on_conflict == "last":
                    # Exclude self's flattened dimensions for this nested space
                    self_prefixes_to_exclude.update(
                        self._get_nested_space_prefixes(
                            parent_name, self.nested_spaces[parent_name]
                        )
                    )
                elif on_conflict == "first":
                    # Exclude other's flattened dimensions for this nested space
                    other_prefixes_to_exclude.update(
                        self._get_nested_space_prefixes(
                            parent_name, other.nested_spaces[parent_name]
                        )
                    )
                elif on_conflict == "error":
                    raise ValueError(f"Conflicting nested space: {parent_name}")

        def should_exclude_self(name: str) -> bool:
            return any(name.startswith(p) for p in self_prefixes_to_exclude)

        def should_exclude_other(name: str) -> bool:
            return any(name.startswith(p) for p in other_prefixes_to_exclude)

        # Copy dimensions from self (excluding those from conflicting nested spaces)
        for name, dim in self.dimensions.items():
            if not should_exclude_self(name):
                result.dimensions[name] = dim

        # Add/merge dimensions from other
        for name, dim in other.dimensions.items():
            # Skip dimensions from nested spaces that should be excluded
            if should_exclude_other(name):
                continue

            if name in result.dimensions:
                existing_dim = result.dimensions[name]

                # Check for type change - raise by default
                if not allow_type_change and existing_dim.dim_type != dim.dim_type:
                    raise ValueError(
                        f"Parameter '{name}' has conflicting types: "
                        f"{existing_dim.dim_type.value} vs {dim.dim_type.value}. "
                        f"Use allow_type_change=True to override."
                    )

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

        # Merge conditions, excluding those targeting excluded dimensions
        result.conditions = [
            c for c in self.conditions if not should_exclude_self(c.target_param)
        ] + [c for c in other.conditions if not should_exclude_other(c.target_param)]

        # Merge constraints (constraints don't have target_param, keep all)
        result.constraints = self.constraints.copy() + other.constraints.copy()

        # Merge nested spaces according to on_conflict strategy
        if on_conflict == "last":
            result.nested_spaces = {**self.nested_spaces, **other.nested_spaces}
        elif on_conflict == "first":
            result.nested_spaces = {**other.nested_spaces, **self.nested_spaces}
        else:  # "error" - conflicts already raised above
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

    def wrap_params(self, flat_params: dict) -> "ParamsView":
        """Wrap flat optimizer params in a ParamsView.

        ParamsView provides transparent access to nested parameter structures
        through the Hybrid NestedValue pattern. Parent parameters in nested
        spaces return NestedValue objects that support parameter access and
        auto-instantiation.

        Parameters
        ----------
        flat_params : dict[str, Any]
            Flat parameters from the optimizer.

        Returns
        -------
        ParamsView
            Smart view providing transparent nested param access.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.svm import SVC

        >>> space = SearchSpace(
        ...     estimator={
        ...         RandomForestClassifier: {"n_estimators": [10, 50, 100]},
        ...         SVC: {"C": (0.1, 10.0, "log")},
        ...     },
        ... )

        >>> flat = {
        ...     "estimator": RandomForestClassifier,
        ...     "randomforestclassifier__n_estimators": 100,
        ... }
        >>> params = space.wrap_params(flat)

        Access nested params:

        >>> params["estimator"]["n_estimators"]
        100

        Auto-instantiate with all params:

        >>> model = params["estimator"]()

        Instantiate with overrides:

        >>> model = params["estimator"](n_jobs=-1)

        Comparison still works:

        >>> params["estimator"] == RandomForestClassifier
        True

        Get raw class if needed:

        >>> params["estimator"].value
        <class 'sklearn.ensemble.RandomForestClassifier'>
        """
        from ._params_view import NestedSpaceConfig, ParamsView

        configs = tuple(
            NestedSpaceConfig(parent_name=name) for name in self.nested_spaces.keys()
        )

        return ParamsView(
            flat_params=flat_params,
            nested_configs=configs,
            prefix_maker=make_prefix,
        )

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
