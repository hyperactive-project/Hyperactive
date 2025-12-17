"""Condition class for conditional search space dimensions.

This module provides the Condition class which specifies when a dimension
is active based on the values of other parameters.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Condition:
    """Specifies when a dimension is active in the search space.

    A condition controls whether a parameter should be sampled based on
    the values of other parameters. This enables conditional/hierarchical
    search spaces.

    Parameters
    ----------
    target_param : str
        The name of the parameter this condition controls.
    predicate : callable
        A function that takes a params dict and returns True if the
        target parameter should be active (sampled).
    depends_on : list of str, default=[]
        Names of parameters this condition depends on. Used for
        optimization and validation purposes.
    name : str or None, default=None
        Optional name for the condition (for debugging/logging).

    Examples
    --------
    >>> # gamma is only active when kernel != "linear"
    >>> condition = Condition(
    ...     target_param="gamma",
    ...     predicate=lambda p: p["kernel"] != "linear",
    ...     depends_on=["kernel"],
    ... )
    >>> condition.is_active({"kernel": "rbf"})
    True
    >>> condition.is_active({"kernel": "linear"})
    False

    >>> # degree is only active when kernel == "poly"
    >>> condition = Condition(
    ...     target_param="degree",
    ...     predicate=lambda p: p["kernel"] == "poly",
    ...     depends_on=["kernel"],
    ... )
    """

    target_param: str
    predicate: Callable[[dict], bool]
    depends_on: list[str] = field(default_factory=list)
    name: str | None = None

    def __post_init__(self):
        """Validate condition after initialization."""
        if not callable(self.predicate):
            raise TypeError(
                f"Condition predicate must be callable, got {type(self.predicate)}"
            )
        if isinstance(self.depends_on, str):
            self.depends_on = [self.depends_on]

    def is_active(self, params: dict) -> bool:
        """Check if the target parameter should be active given current params.

        Parameters
        ----------
        params : dict
            The current parameter values.

        Returns
        -------
        bool
            True if the target parameter should be sampled/active.
        """
        try:
            return bool(self.predicate(params))
        except KeyError:
            # If a dependency is missing, consider the condition inactive
            return False

    def can_evaluate(self, params: dict) -> bool:
        """Check if all dependencies are available to evaluate this condition.

        Parameters
        ----------
        params : dict
            The current parameter values.

        Returns
        -------
        bool
            True if all dependencies are present in params.
        """
        return all(dep in params for dep in self.depends_on)

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f"name={self.name!r}, " if self.name else ""
        return (
            f"Condition({name_str}target={self.target_param!r}, "
            f"depends_on={self.depends_on})"
        )
