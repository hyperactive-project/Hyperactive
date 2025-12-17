"""ParamsView and NestedValue for transparent nested search space access.

This module provides the ParamsView class that wraps flat optimizer parameters
and provides transparent access to nested parameter structures through the
Hybrid NestedValue pattern.

Key Features:
- params["estimator"]["max_depth"] -> nested param access
- params["estimator"]() -> auto-instantiate with all params
- params["estimator"] == SomeClass -> comparison still works
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

__all__ = ["ParamsView", "NestedValue", "NestedSpaceConfig", "create_params_view"]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class PrefixMaker(Protocol):
    """Protocol for creating prefixes from parent values."""

    def __call__(self, value: Any) -> str:
        """Create a prefix string from a parent value."""
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedSpaceConfig:
    """Configuration for a single nested space.

    Parameters
    ----------
    parent_name : str
        The name of the parent parameter (e.g., "estimator").
    """

    parent_name: str


# ---------------------------------------------------------------------------
# NestedValue
# ---------------------------------------------------------------------------


class NestedValue:
    """Hybrid object that acts as both a value and a parameter namespace.

    NestedValue wraps a parent value (typically a class or callable) along
    with its associated parameters. It provides multiple access patterns:

    1. **Subscript access**: Get individual parameters
       >>> nested["max_depth"]
       10

    2. **Call to instantiate**: Create instance with stored params
       >>> nested()
       RandomForestClassifier(n_estimators=100, max_depth=10)

    3. **Call with overrides**: Override or add parameters
       >>> nested(n_jobs=-1)
       RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)

    4. **Comparison**: Compare against the wrapped value
       >>> nested == RandomForestClassifier
       True

    5. **Iteration**: Iterate over parameter names
       >>> list(nested)
       ['n_estimators', 'max_depth']

    Parameters
    ----------
    value : Any
        The wrapped value (class, function, or any object).
    params : dict[str, Any]
        Parameters associated with this value.

    Attributes
    ----------
    value : Any
        The unwrapped original value.
    params : dict[str, Any]
        Copy of the associated parameters.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> nested = NestedValue(
    ...     value=RandomForestClassifier,
    ...     params={"n_estimators": 100, "max_depth": 10},
    ... )

    Access parameters:

    >>> nested["n_estimators"]
    100

    Instantiate with all parameters:

    >>> model = nested()
    >>> model.n_estimators
    100

    Override parameters:

    >>> model = nested(n_estimators=200, n_jobs=-1)
    >>> model.n_estimators
    200

    Compare to original class:

    >>> nested == RandomForestClassifier
    True
    """

    __slots__ = ("_value", "_params")

    def __init__(self, value: Any, params: dict[str, Any]):
        self._value = value
        self._params = dict(params)  # Defensive copy

    # -------------------------------------------------------------------------
    # Core access patterns
    # -------------------------------------------------------------------------

    @property
    def value(self) -> Any:
        """The unwrapped original value (class, function, etc.)."""
        return self._value

    @property
    def params(self) -> dict[str, Any]:
        """Copy of the associated parameters."""
        return dict(self._params)

    def __getitem__(self, key: str) -> Any:
        """Get a parameter by name.

        Parameters
        ----------
        key : str
            Parameter name.

        Returns
        -------
        Any
            Parameter value.

        Raises
        ------
        KeyError
            If parameter not found.
        """
        return self._params[key]

    def __call__(self, **overrides) -> Any:
        """Instantiate/call the wrapped value with stored params.

        Parameters
        ----------
        **overrides
            Additional parameters or overrides for stored params.

        Returns
        -------
        Any
            Result of calling the wrapped value with merged parameters.

        Examples
        --------
        >>> nested()  # Use stored params only
        >>> nested(n_jobs=-1)  # Add n_jobs
        >>> nested(max_depth=20)  # Override max_depth
        """
        merged = {**self._params, **overrides}
        return self._value(**merged)

    # -------------------------------------------------------------------------
    # Mapping-like interface
    # -------------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._params)

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._params)

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with default."""
        return self._params.get(key, default)

    def keys(self):
        """Parameter names."""
        return self._params.keys()

    def values(self):
        """Parameter values."""
        return self._params.values()

    def items(self):
        """Parameter items."""
        return self._params.items()

    # -------------------------------------------------------------------------
    # Comparison - delegate to wrapped value
    # -------------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        """Compare to wrapped value or another NestedValue."""
        if isinstance(other, NestedValue):
            return self._value == other._value and self._params == other._params
        return self._value == other

    def __ne__(self, other: Any) -> bool:
        """Not equal comparison."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Hash based on wrapped value."""
        return hash(self._value)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Detailed representation."""
        value_repr = getattr(self._value, "__name__", repr(self._value))
        return f"NestedValue({value_repr}, params={self._params})"

    def __str__(self) -> str:
        """User-friendly string showing the wrapped value."""
        return str(getattr(self._value, "__name__", self._value))


# ---------------------------------------------------------------------------
# ParamsView
# ---------------------------------------------------------------------------


class ParamsView(Mapping[str, Any]):
    """Immutable, dict-like view over optimization parameters.

    Provides transparent access to nested parameter structures through
    the Hybrid NestedValue pattern. Parent parameters in nested spaces
    return NestedValue objects that support both parameter access and
    auto-instantiation.

    This class implements ``collections.abc.Mapping``, making it usable
    anywhere a read-only dict-like object is expected.

    Parameters
    ----------
    flat_params : dict[str, Any]
        The flat parameter dictionary from the optimizer.
    nested_configs : tuple[NestedSpaceConfig, ...]
        Configuration for each nested space. Empty tuple if no nesting.
    prefix_maker : callable
        Callable that creates prefixes from parent values.
        Signature: (parent_value: Any) -> str

    Notes
    -----
    **Key behavior for nested spaces:**

    When a parameter is configured as a nested space parent (e.g., "estimator"),
    accessing it returns a NestedValue object instead of the raw value:

    >>> params["estimator"]
    NestedValue(RandomForestClassifier, params={'n_estimators': 100, ...})

    This NestedValue supports multiple access patterns:

    >>> params["estimator"]["max_depth"]      # Get nested param
    >>> params["estimator"]()                 # Instantiate with all params
    >>> params["estimator"](n_jobs=-1)        # Instantiate with override
    >>> params["estimator"] == RFC            # Comparison works
    >>> params["estimator"].value             # Get raw class
    >>> params["estimator"].params            # Get params dict

    **For non-nested parameters:**

    Regular parameters return their values directly:

    >>> params["learning_rate"]
    0.01

    **Iteration:**

    Iteration yields logical keys (parent names, not prefixed children).
    Prefixed keys are hidden but still accessible directly.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> config = NestedSpaceConfig(parent_name="estimator")
    >>> view = ParamsView(
    ...     flat_params={
    ...         "estimator": RandomForestClassifier,
    ...         "randomforestclassifier__n_estimators": 100,
    ...         "randomforestclassifier__max_depth": 10,
    ...         "learning_rate": 0.01,
    ...     },
    ...     nested_configs=(config,),
    ...     prefix_maker=lambda cls: cls.__name__.lower(),
    ... )

    Access nested params:

    >>> view["estimator"]["n_estimators"]
    100

    Auto-instantiate:

    >>> model = view["estimator"]()
    >>> model.n_estimators
    100

    Regular param access:

    >>> view["learning_rate"]
    0.01
    """

    __slots__ = (
        "_flat",
        "_nested_configs",
        "_prefix_maker",
        "_nested_cache",
        "_nested_value_cache",
        "_logical_keys",
        "_nested_parent_names",
    )

    def __init__(
        self,
        flat_params: dict[str, Any],
        nested_configs: tuple[NestedSpaceConfig, ...] = (),
        prefix_maker: PrefixMaker | None = None,
    ):
        self._flat = flat_params
        self._nested_configs = nested_configs
        self._prefix_maker = prefix_maker or _default_prefix_maker
        self._nested_cache: dict[str, dict[str, Any]] = {}
        self._nested_value_cache: dict[str, NestedValue] = {}

        # Pre-compute for fast lookups
        self._nested_parent_names: frozenset[str] = frozenset(
            c.parent_name for c in nested_configs
        )
        self._logical_keys: tuple[str, ...] = self._compute_logical_keys()

    def _compute_logical_keys(self) -> tuple[str, ...]:
        """Compute the logical keys visible to users.

        Logical keys are non-prefixed keys from flat_params.
        Prefixed keys (containing "__") are excluded from iteration
        when nested spaces exist, but remain accessible directly.
        """
        keys = []

        # When we have nested spaces, hide ALL keys containing "__"
        # This matches sklearn's convention where "__" is the nested separator
        has_nested = bool(self._nested_configs)

        for key in self._flat:
            if has_nested and "__" in key:
                # Hide prefixed keys from iteration
                continue
            keys.append(key)

        return tuple(keys)

    # -------------------------------------------------------------------------
    # Mapping interface
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Get a parameter value by key.

        For nested parent keys, returns a NestedValue object.
        For regular keys, returns the value directly.

        Parameters
        ----------
        key : str
            Parameter name.

        Returns
        -------
        Any
            The parameter value, or NestedValue for nested parents.

        Raises
        ------
        KeyError
            If key is not found.
        """
        # Check if this is a nested parent
        if key in self._nested_parent_names and key in self._flat:
            return self._get_nested_value(key)

        # Direct access to flat params (includes prefixed keys)
        if key in self._flat:
            return self._flat[key]

        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over logical keys.

        Yields non-prefixed keys. Prefixed keys are hidden from iteration.
        """
        return iter(self._logical_keys)

    def __len__(self) -> int:
        """Return count of logical keys."""
        return len(self._logical_keys)

    def __contains__(self, key: object) -> bool:
        """Check if key exists (logical or prefixed)."""
        if not isinstance(key, str):
            return False
        return key in self._flat

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    # -------------------------------------------------------------------------
    # NestedValue creation
    # -------------------------------------------------------------------------

    def _get_nested_value(self, parent_name: str) -> NestedValue:
        """Get or create NestedValue for a parent parameter."""
        if parent_name in self._nested_value_cache:
            return self._nested_value_cache[parent_name]

        parent_value = self._flat[parent_name]
        nested_params = self._extract_nested_params(parent_name, parent_value)

        nested_value = NestedValue(value=parent_value, params=nested_params)
        self._nested_value_cache[parent_name] = nested_value

        return nested_value

    def _extract_nested_params(
        self, parent_name: str, parent_value: Any
    ) -> dict[str, Any]:
        """Extract nested params for a parent value."""
        if parent_name in self._nested_cache:
            return self._nested_cache[parent_name]

        prefix = self._prefix_maker(parent_value) + "__"
        prefix_len = len(prefix)

        nested = {
            key[prefix_len:]: value
            for key, value in self._flat.items()
            if key.startswith(prefix)
        }

        self._nested_cache[parent_name] = nested
        return nested

    # -------------------------------------------------------------------------
    # Conversion methods
    # -------------------------------------------------------------------------

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a copy of the original flat dictionary.

        This is the raw format from the optimizer, including all
        prefixed keys. Nested parents are raw values, not NestedValue.

        Returns
        -------
        dict[str, Any]
            Copy of flat parameters.
        """
        return dict(self._flat)

    def to_nested_dict(self) -> dict[str, Any]:
        """Return a restructured dictionary with nested params grouped.

        Nested parents include a ``{parent}_params`` key with extracted params.
        Prefixed keys are removed.

        Returns
        -------
        dict[str, Any]
            Restructured parameters.

        Examples
        --------
        >>> view.to_nested_dict()
        {
            'estimator': RandomForestClassifier,
            'estimator_params': {'n_estimators': 100, 'max_depth': 10},
            'learning_rate': 0.01,
        }
        """
        result = {}
        processed_prefixes = set()

        # Handle nested parents
        for parent_name in self._nested_parent_names:
            if parent_name in self._flat:
                parent_value = self._flat[parent_name]
                result[parent_name] = parent_value
                result[f"{parent_name}_params"] = self._extract_nested_params(
                    parent_name, parent_value
                )
                processed_prefixes.add(self._prefix_maker(parent_value) + "__")

        # Add non-nested, non-prefixed params
        for key, value in self._flat.items():
            if key not in result:
                is_prefixed = any(key.startswith(p) for p in processed_prefixes)
                if not is_prefixed:
                    result[key] = value

        return result

    def raw(self, key: str) -> Any:
        """Get raw value without NestedValue wrapping.

        Useful when you need the actual class/callable, not wrapped.

        Parameters
        ----------
        key : str
            Parameter name.

        Returns
        -------
        Any
            Raw value from flat params.
        """
        return self._flat[key]

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return string representation showing logical view."""
        items = []
        for k in self._logical_keys:
            v = self[k]
            items.append(f"{k!r}: {v!r}")
        return f"ParamsView({{{', '.join(items)}}})"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return str(self.to_nested_dict())


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_params_view(
    flat_params: dict[str, Any],
    nested_space_names: set[str],
    prefix_maker: PrefixMaker,
) -> ParamsView:
    """Create a ParamsView from SearchSpace information.

    This function bridges SearchSpace and ParamsView, extracting only
    the information ParamsView needs.

    Parameters
    ----------
    flat_params : dict[str, Any]
        Flat parameters from optimizer.
    nested_space_names : set[str]
        Names of nested spaces (e.g., {"estimator", "scaler"}).
    prefix_maker : callable
        Function to create prefixes from parent values.

    Returns
    -------
    ParamsView
        Configured view over the parameters.
    """
    configs = tuple(NestedSpaceConfig(parent_name=name) for name in nested_space_names)

    return ParamsView(
        flat_params=flat_params,
        nested_configs=configs,
        prefix_maker=prefix_maker,
    )


# ---------------------------------------------------------------------------
# Default prefix maker
# ---------------------------------------------------------------------------


def _default_prefix_maker(value: Any) -> str:
    """Default prefix maker matching SearchSpace._make_prefix behavior."""
    if isinstance(value, type):
        return value.__name__.lower()
    elif callable(value):
        return getattr(value, "__name__", str(value)).lower()
    return str(value).lower().replace(" ", "_").replace("-", "_")
