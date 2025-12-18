"""Condition manager for conditional search space dimensions.

This module provides the ConditionManager class which handles storage,
validation, and evaluation of conditions that control when parameters are active.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ._condition import Condition

if TYPE_CHECKING:
    from ._dimension_registry import DimensionRegistry

__all__ = ["ConditionManager"]


class ConditionManager:
    """Manages conditions and validates dependency graphs.

    Handles condition storage, circular dependency detection, and
    evaluation of which parameters are active given current values.

    Parameters
    ----------
    dimension_registry : DimensionRegistry
        Reference to dimension registry for parameter validation.

    Examples
    --------
    >>> from hyperactive.search_space import DimensionRegistry
    >>> registry = DimensionRegistry()
    >>> registry.add("kernel", ["rbf", "linear", "poly"])
    >>> registry.add("gamma", (0.01, 10.0))
    >>> manager = ConditionManager(registry)
    >>> manager.add("gamma", when=lambda p: p["kernel"] != "linear")
    >>> manager.filter_active({"kernel": "rbf", "gamma": 1.0})
    {'kernel': 'rbf', 'gamma': 1.0}
    >>> manager.filter_active({"kernel": "linear", "gamma": 1.0})
    {'kernel': 'linear'}
    """

    def __init__(self, dimension_registry: "DimensionRegistry"):
        """Initialize condition manager.

        Parameters
        ----------
        dimension_registry : DimensionRegistry
            Reference to dimension registry for parameter validation.
        """
        self._conditions: list[Condition] = []
        self._dimension_registry = dimension_registry

    def add(
        self,
        param: str | None = None,
        when: Callable[[dict], bool] | None = None,
        depends_on: str | list[str] | None = None,
        name: str | None = None,
        *,
        condition: Condition | None = None,
        skip_validation: bool = False,
    ) -> None:
        """Add a condition for when a parameter is active.

        Can be called in two ways:

        1. With individual parameters: ``add(param="x", when=lambda p: ...)``
        2. With pre-built condition: ``add(condition=cond, skip_validation=True)``

        Parameters
        ----------
        param : str, optional
            Name of the parameter this condition controls.
        when : callable, optional
            Function that takes params dict and returns True if param is active.
        depends_on : str or list[str], optional
            Parameter(s) this condition depends on (for optimization).
        name : str, optional
            Optional name for the condition.
        condition : Condition, optional
            Pre-constructed Condition object (alternative to individual params).
        skip_validation : bool, default=False
            If True, skips parameter existence and circular dependency checks.
            Only use for internally-generated conditions (e.g., from nested spaces).

        Raises
        ------
        ValueError
            If validation fails (unknown param, circular dependency).
        """
        # Build condition from params or use provided
        if condition is not None:
            new_condition = condition
        else:
            if param is None or when is None:
                raise ValueError("Must provide either (param, when) or condition")

            if depends_on is None:
                depends_on = []
            elif isinstance(depends_on, str):
                depends_on = [depends_on]

            new_condition = Condition(
                target_param=param,
                predicate=when,
                depends_on=depends_on,
                name=name,
            )

        # Validation (unless explicitly skipped)
        if not skip_validation:
            if new_condition.target_param not in self._dimension_registry:
                raise ValueError(
                    f"Unknown parameter: {new_condition.target_param}. "
                    f"Available parameters: {list(self._dimension_registry.keys())}"
                )

            unknown_deps = (
                set(new_condition.depends_on) - set(self._dimension_registry.keys())
            )
            if unknown_deps:
                raise ValueError(
                    f"Unknown parameters in depends_on: {unknown_deps}. "
                    f"Available parameters: {list(self._dimension_registry.keys())}"
                )

            # Check for circular dependencies BEFORE adding the condition.
            # This ensures the manager remains in a valid state if the check fails.
            self._check_circular_dependencies(new_condition)

        self._conditions.append(new_condition)

    def merge_conditions(self, conditions: list[Condition]) -> None:
        """Merge conditions from another source (e.g., during union).

        Adds all conditions without validation. Use this for merging
        already-validated conditions from another SearchSpace.

        Parameters
        ----------
        conditions : list[Condition]
            The conditions to merge.
        """
        self._conditions.extend(conditions)

    @property
    def conditions(self) -> list[Condition]:
        """Get the list of conditions.

        Returns the actual internal list for backward compatibility.
        """
        return self._conditions

    @property
    def has_conditions(self) -> bool:
        """Check if there are any conditions.

        Returns
        -------
        bool
            True if there are conditions.
        """
        return len(self._conditions) > 0

    def filter_active(self, params: dict) -> dict:
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
            if name not in self._dimension_registry:
                continue

            # Check if any condition disables this parameter
            is_active = True
            for condition in self._conditions:
                if condition.target_param == name:
                    if condition.can_evaluate(params) and not condition.is_active(
                        params
                    ):
                        is_active = False
                        break

            if is_active:
                active_params[name] = value

        return active_params

    def _check_circular_dependencies(
        self, new_condition: Condition | None = None
    ) -> None:
        """Check for circular dependencies in conditions.

        Builds a dependency graph from conditions and detects cycles using DFS.

        Parameters
        ----------
        new_condition : Condition, optional
            A new condition to include in the check (not yet added to conditions).
            This allows validating before adding to ensure consistent state.

        Raises
        ------
        ValueError
            If circular dependencies are detected.
        """
        # Build adjacency list: param -> list of params it depends on
        # Note: A condition on param P with depends_on=[D1, D2] means
        # P depends on D1 and D2 (edges: P->D1, P->D2)
        graph: dict[str, set[str]] = {}

        # Include the new condition in the check if provided
        conditions_to_check = list(self._conditions)
        if new_condition is not None:
            conditions_to_check.append(new_condition)

        for condition in conditions_to_check:
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

    def __len__(self) -> int:
        """Return number of conditions."""
        return len(self._conditions)

    def __iter__(self):
        """Iterate over conditions."""
        return iter(self._conditions)

    def copy(self) -> "ConditionManager":
        """Create a copy of this manager.

        Note: Creates a shallow copy that shares the same dimension registry.

        Returns
        -------
        ConditionManager
            New manager with copies of all conditions.
        """
        new_manager = ConditionManager(self._dimension_registry)
        new_manager._conditions = self._conditions.copy()
        return new_manager
