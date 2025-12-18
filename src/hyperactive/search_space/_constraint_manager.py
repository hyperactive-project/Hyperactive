"""Constraint manager for search space constraints.

This module provides the ConstraintManager class which handles storage
and evaluation of constraints on parameter combinations.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Callable

from ._constraint import Constraint

__all__ = ["ConstraintManager"]


class ConstraintManager:
    """Manages constraint storage and evaluation.

    Constraints are predicates that filter out invalid parameter combinations.
    This class provides a focused interface for adding and evaluating constraints.

    Examples
    --------
    >>> manager = ConstraintManager()
    >>> manager.add(lambda p: p["x"] + p["y"] < 10, name="sum_limit")
    >>> manager.check_all({"x": 3, "y": 5})
    True
    >>> manager.check_all({"x": 7, "y": 5})
    False
    """

    def __init__(self):
        """Initialize empty constraint manager."""
        self._constraints: list[Constraint] = []

    def add(
        self,
        predicate: Callable[[dict], bool],
        name: str | None = None,
        params: list[str] | None = None,
    ) -> None:
        """Add a constraint.

        Parameters
        ----------
        predicate : callable
            Function that takes params dict and returns True if valid.
        name : str, optional
            Name for the constraint (for debugging).
        params : list[str], optional
            Parameter names involved in this constraint.
        """
        if name is None:
            name = f"constraint_{len(self._constraints)}"

        self._constraints.append(
            Constraint(
                predicate=predicate,
                name=name,
                params=params,
            )
        )

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a pre-constructed Constraint object.

        Parameters
        ----------
        constraint : Constraint
            The constraint to add.
        """
        self._constraints.append(constraint)

    @property
    def constraints(self) -> list[Constraint]:
        """Get the list of constraints.

        Returns the actual internal list for backward compatibility.
        """
        return self._constraints

    def check_all(self, params: dict) -> bool:
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
        return all(c.is_satisfied(params) for c in self._constraints)

    @property
    def has_constraints(self) -> bool:
        """Check if there are any constraints.

        Returns
        -------
        bool
            True if there are constraints.
        """
        return len(self._constraints) > 0

    def __len__(self) -> int:
        """Return number of constraints."""
        return len(self._constraints)

    def __iter__(self):
        """Iterate over constraints."""
        return iter(self._constraints)

    def copy(self) -> "ConstraintManager":
        """Create a copy of this manager with copied constraints.

        Returns
        -------
        ConstraintManager
            New manager with copies of all constraints.
        """
        new_manager = ConstraintManager()
        new_manager._constraints = self._constraints.copy()
        return new_manager
