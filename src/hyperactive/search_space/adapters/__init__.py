"""Search space adapters for different backends.

This module provides adapters that convert SearchSpace objects to
backend-specific formats for GFO, Optuna, and sklearn.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._base import BaseSearchSpaceAdapter
from ._gfo import GFOSearchSpaceAdapter
from ._optuna import OptunaSearchSpaceAdapter
from ._sklearn import SklearnSearchSpaceAdapter

__all__ = [
    "BaseSearchSpaceAdapter",
    "GFOSearchSpaceAdapter",
    "OptunaSearchSpaceAdapter",
    "SklearnSearchSpaceAdapter",
    "get_adapter",
]


def get_adapter(backend: str, search_space, **kwargs):
    """Get the appropriate adapter for a backend.

    Parameters
    ----------
    backend : str
        The backend name: "gfo", "optuna", "sklearn_grid", or "sklearn_random".
    search_space : SearchSpace
        The search space to adapt.
    **kwargs
        Additional arguments passed to the adapter.

    Returns
    -------
    BaseSearchSpaceAdapter
        The appropriate adapter instance.

    Raises
    ------
    ValueError
        If the backend is not recognized.

    Examples
    --------
    >>> from hyperactive.search_space import SearchSpace
    >>> from hyperactive.search_space.adapters import get_adapter
    >>> space = SearchSpace(x=(0.0, 10.0), y=["a", "b", "c"])
    >>> adapter = get_adapter("gfo", space, resolution=50)
    >>> gfo_space = adapter.adapt()
    """
    backend = backend.lower()

    if backend == "gfo":
        return GFOSearchSpaceAdapter(search_space, **kwargs)
    elif backend == "optuna":
        return OptunaSearchSpaceAdapter(search_space)
    elif backend in ("sklearn", "sklearn_random", "sklearn_grid"):
        return SklearnSearchSpaceAdapter(search_space)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Expected 'gfo', 'optuna', 'sklearn', 'sklearn_random', or 'sklearn_grid'."
        )
