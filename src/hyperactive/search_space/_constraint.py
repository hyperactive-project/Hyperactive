"""Constraint class for search space constraints.

This module provides the Constraint class which defines predicates
that filter out invalid parameter combinations.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Constraint:
    """Defines a constraint on the search space.

    A constraint is a predicate that must return True for a parameter
    combination to be valid. Invalid combinations will be filtered out
    or penalized during optimization.

    Parameters
    ----------
    predicate : callable
        A function that takes a params dict and returns True if the
        combination is valid.
    name : str or None, default=None
        Optional name for the constraint (for debugging/logging).
    params : list of str or None, default=None
        Optional list of parameter names involved in this constraint.
        Used for optimization and validation purposes.

    Examples
    --------
    >>> # batch_size * lr must be less than 0.5
    >>> constraint = Constraint(
    ...     predicate=lambda p: p["batch_size"] * p["lr"] < 0.5,
    ...     name="batch_lr_limit",
    ... )
    >>> constraint.is_satisfied({"batch_size": 32, "lr": 0.01})
    True
    >>> constraint.is_satisfied({"batch_size": 256, "lr": 0.1})
    False

    >>> # model size constraint
    >>> constraint = Constraint(
    ...     predicate=lambda p: p["n_layers"] * p["hidden_dim"] < 4096,
    ...     name="model_size_limit",
    ...     params=["n_layers", "hidden_dim"],
    ... )
    """

    predicate: Callable[[dict], bool]
    name: str | None = None
    params: list[str] | None = None

    def __post_init__(self):
        """Validate constraint after initialization."""
        if not callable(self.predicate):
            raise TypeError(
                f"Constraint predicate must be callable, got {type(self.predicate)}"
            )

    def is_satisfied(self, params: dict) -> bool:
        """Check if the constraint is satisfied for given parameters.

        Parameters
        ----------
        params : dict
            The parameter values to check.

        Returns
        -------
        bool
            True if the constraint is satisfied.

        Raises
        ------
        KeyError
            If a required parameter is missing from params.
        TypeError
            If parameters have incompatible types for the constraint.
        RuntimeError
            If the predicate raises an unexpected error.
        """
        try:
            return bool(self.predicate(params))
        except KeyError as e:
            constraint_name = self.name or "unnamed constraint"
            params_info = f" (declared params: {self.params})" if self.params else ""
            raise KeyError(
                f"Constraint '{constraint_name}' failed: missing parameter {e}. "
                f"Available parameters: {list(params.keys())}{params_info}"
            ) from e
        except TypeError as e:
            constraint_name = self.name or "unnamed constraint"
            raise TypeError(
                f"Constraint '{constraint_name}' failed with TypeError: {e}. "
                f"Check that parameter types are compatible. Parameters: {params}"
            ) from e
        except Exception as e:
            constraint_name = self.name or "unnamed constraint"
            raise RuntimeError(
                f"Constraint '{constraint_name}' raised {type(e).__name__}: {e}. "
                f"Parameters: {params}"
            ) from e

    def __call__(self, params: dict) -> bool:
        """Allow constraint to be called directly like a function.

        Parameters
        ----------
        params : dict
            The parameter values to check.

        Returns
        -------
        bool
            True if the constraint is satisfied.
        """
        return self.is_satisfied(params)

    def __repr__(self) -> str:
        """Return string representation."""
        if self.name:
            return f"Constraint(name={self.name!r})"
        return f"Constraint(predicate={self.predicate})"
