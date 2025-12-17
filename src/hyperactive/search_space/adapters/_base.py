"""Base adapter for search space conversion.

This module provides the base class for search space adapters that
convert SearchSpace objects to backend-specific formats.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._search_space import SearchSpace


class BaseSearchSpaceAdapter(ABC):
    """Base class for search space adapters.

    Adapters convert SearchSpace objects to backend-specific formats.
    Each backend (GFO, Optuna, sklearn) has its own adapter subclass.

    Parameters
    ----------
    search_space : SearchSpace
        The search space to adapt.

    Attributes
    ----------
    space : SearchSpace
        The search space being adapted.
    """

    def __init__(self, search_space: SearchSpace):
        """Initialize the adapter.

        Parameters
        ----------
        search_space : SearchSpace
            The search space to adapt.
        """
        self.space = search_space

    @abstractmethod
    def adapt(self, **kwargs) -> Any:
        """Convert the search space to backend-specific format.

        Parameters
        ----------
        **kwargs
            Backend-specific options.

        Returns
        -------
        Any
            The backend-specific search space representation.
        """
        raise NotImplementedError("Subclasses must implement adapt()")

    @abstractmethod
    def get_constraints(self) -> list:
        """Get constraints in backend-specific format.

        Returns
        -------
        list
            List of constraints in backend-specific format.
        """
        raise NotImplementedError("Subclasses must implement get_constraints()")
