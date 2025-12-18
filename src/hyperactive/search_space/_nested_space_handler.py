"""Nested space handler for hierarchical search spaces.

This module provides the NestedSpaceHandler class which manages detection,
validation, and expansion of nested search spaces.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._condition import Condition
from ._dimension import Dimension, DimensionType, make_prefix

if TYPE_CHECKING:
    from ._condition_manager import ConditionManager
    from ._dimension_registry import DimensionRegistry
    from ._search_space import SearchSpace

__all__ = ["NestedSpaceHandler"]


class NestedSpaceHandler:
    """Handles nested search space detection, validation, and expansion.

    Nested spaces allow hierarchical parameter structures where child
    parameters are only active when a parent parameter has a specific value.

    Parameters
    ----------
    dimension_registry : DimensionRegistry
        Reference to dimension registry for adding expanded dimensions.
    condition_manager : ConditionManager
        Reference to condition manager for adding auto-generated conditions.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> handler = NestedSpaceHandler(dim_registry, cond_manager)
    >>> handler.add("estimator", {
    ...     RandomForestClassifier: {"n_estimators": [10, 50, 100]},
    ...     SVC: {"C": (0.01, 100.0, "log")},
    ... })
    """

    def __init__(
        self,
        dimension_registry: "DimensionRegistry",
        condition_manager: "ConditionManager",
    ):
        """Initialize nested space handler.

        Parameters
        ----------
        dimension_registry : DimensionRegistry
            Reference to dimension registry.
        condition_manager : ConditionManager
            Reference to condition manager.
        """
        self._dimension_registry = dimension_registry
        self._condition_manager = condition_manager
        self._nested_spaces: dict[str, dict[Any, "SearchSpace"]] = {}

    @staticmethod
    def looks_like_nested_space(value: dict) -> bool:
        """Check if a dict looks like a nested search space definition.

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

    def add(self, name: str, nested: dict[Any, dict]) -> None:
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

        # Convert nested dicts to SearchSpaces (lazy import to avoid circular)
        from ._search_space import SearchSpace

        converted: dict[Any, SearchSpace] = {}
        for key, subspace_dict in nested.items():
            converted[key] = SearchSpace(subspace_dict)

        self._nested_spaces[name] = converted

        # Create parent categorical dimension from keys
        parent_values = list(nested.keys())
        self._dimension_registry.add_dimension(
            Dimension(
                name=name,
                values=parent_values,
                dim_type=DimensionType.CATEGORICAL,
                dtype=object,
            )
        )

        # Expand nested space to flat dimensions with conditions
        self._expand_nested_space(name, converted)

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
                collision_details.append(
                    f"  prefix '{prefix}__': {', '.join(key_reprs)}"
                )

            raise ValueError(
                f"Nested space '{name}' has keys that produce the same prefix, "
                f"which would cause parameter name collisions:\n"
                + "\n".join(collision_details)
                + "\n\nUse named functions or classes instead of lambdas, "
                "or ensure all keys have unique names."
            )

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
                self._dimension_registry.add_dimension(new_dim)

                # Add condition: only active when parent == this value
                # Use default argument to capture parent_value correctly in closure
                self._condition_manager.add_condition(
                    Condition(
                        target_param=flat_name,
                        predicate=lambda p, pn=parent_name, pv=parent_value: p.get(pn)
                        == pv,
                        depends_on=[parent_name],
                        name=f"{flat_name}_when_{parent_name}=={make_prefix(parent_value)}",
                    )
                )

    @property
    def nested_spaces(self) -> dict[str, dict[Any, "SearchSpace"]]:
        """Get the nested spaces dictionary.

        Returns the actual internal dict for backward compatibility.
        """
        return self._nested_spaces

    @property
    def has_nested_spaces(self) -> bool:
        """Check if there are any nested spaces.

        Returns
        -------
        bool
            True if there are nested spaces.
        """
        return len(self._nested_spaces) > 0

    def __len__(self) -> int:
        """Return number of nested spaces."""
        return len(self._nested_spaces)

    def copy(self) -> "NestedSpaceHandler":
        """Create a shallow copy of this handler.

        Note: Shares the same dimension registry and condition manager.
        The nested_spaces dict is copied but SearchSpace objects are shared.

        Returns
        -------
        NestedSpaceHandler
            New handler with copied nested spaces dict.
        """
        new_handler = NestedSpaceHandler(
            self._dimension_registry, self._condition_manager
        )
        new_handler._nested_spaces = {**self._nested_spaces}
        return new_handler
