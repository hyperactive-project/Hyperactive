"""SearchSpace class for unified search space specification.

This module provides the main SearchSpace class that enables a unified
API for specifying search spaces across all Hyperactive optimizers.

SearchSpace is implemented as a facade that coordinates specialized components:
- DimensionRegistry: dimension storage and lookup
- ConditionManager: conditions and circular dependency detection
- ConstraintManager: constraint storage and evaluation
- NestedSpaceHandler: nested search space detection and expansion
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any, Callable

from ._condition import Condition
from ._condition_manager import ConditionManager
from ._constraint import Constraint
from ._constraint_manager import ConstraintManager
from ._dimension import Dimension, DimensionType, make_prefix
from ._dimension_registry import DimensionRegistry
from ._nested_space_handler import NestedSpaceHandler

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
    The dimension type is inferred from the Python type of each value:

    **Categorical (list)**
        Any Python list becomes a categorical dimension. Lists can contain
        any hashable values: strings, numbers, booleans, None, classes, or
        functions. The optimizer will sample from these discrete choices.

        >>> kernel = ["rbf", "linear", "poly"]       # strings
        >>> max_depth = [3, 5, 10, None]             # mixed with None
        >>> estimator = [RandomForest, SVC]         # classes

    **Continuous Integer (tuple of two ints)**
        A tuple of two integers defines a continuous integer range.
        The optimizer samples integers uniformly between low and high.

        >>> n_layers = (1, 10)      # integers from 1 to 10
        >>> batch_size = (16, 256)  # integers from 16 to 256

    **Continuous Float (tuple of two floats)**
        A tuple containing at least one float defines a continuous float range.
        The optimizer samples floats uniformly between low and high.

        >>> dropout = (0.0, 0.5)         # floats from 0.0 to 0.5
        >>> weight_decay = (0.0, 0.1)    # floats from 0.0 to 0.1

    **Log-scale Continuous (tuple with "log")**
        Adding "log" as the third element enables logarithmic sampling.
        Use this for parameters spanning multiple orders of magnitude.
        Both bounds must be positive (log of zero/negative is undefined).

        >>> lr = (1e-5, 1e-1, "log")     # log-uniform from 0.00001 to 0.1
        >>> C = (0.01, 100.0, "log")     # log-uniform from 0.01 to 100

    **Discrete (numpy.ndarray)**
        Numpy arrays define a discrete set of pre-computed values.
        The optimizer samples from this fixed set of values.

        >>> x = np.arange(-10, 10, 0.5)           # fixed grid
        >>> hidden = np.array([32, 64, 128, 256]) # specific values
        >>> rates = np.logspace(-4, -1, 20)       # log-spaced grid

    **Distribution (scipy.stats)**
        Scipy frozen distributions are passed through for backends that
        support them. For GFO, samples are drawn to create a discrete array.

        >>> dropout = scipy.stats.beta(2, 5)      # beta distribution
        >>> noise = scipy.stats.norm(0, 1)        # normal distribution

    **Constant (scalar)**
        Single values (int, float, str, bool, None) become constants that
        are not searched. They are passed through to the objective function.

        >>> seed = 42           # fixed integer
        >>> verbose = False     # fixed boolean

    Nested Search Spaces
    --------------------
    Nested spaces allow hierarchical parameter structures. A dict is
    automatically detected as a nested space when:

    1. ALL keys are classes or callables (not strings/numbers)
    2. ALL values are dicts (parameter specifications)

    This is unambiguous because categorical dimensions use lists, not dicts:

    >>> # Categorical: use a LIST of choices
    >>> transform = [np.log, np.sqrt, np.exp]

    >>> # Nested space: use a DICT with class keys and dict values
    >>> estimator = {
    ...     RandomForest: {"n_estimators": [10, 50, 100]},
    ...     SVC: {"C": (0.01, 100.0, "log")},
    ... }

    When a nested space is detected:

    1. A categorical dimension is created from the dict keys
    2. Child parameters are flattened with prefixes (e.g., "svc__C")
    3. Conditions are auto-generated so child params activate only when
       their parent value is selected

    Conditions vs Constraints
    -------------------------
    **Conditions** control whether a parameter is active (exists in the
    search). Use conditions when a parameter only makes sense for certain
    values of another parameter.

    >>> space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")

    **Constraints** filter out invalid parameter combinations. Use constraints
    when parameters interact in ways that make certain combinations invalid.

    >>> space.add_constraint(lambda p: p["x"] + p["y"] < 10)

    Key difference: conditions affect which parameters appear in a sample,
    while constraints reject entire samples that violate rules.

    Backend Compatibility
    ---------------------
    SearchSpace automatically adapts to different optimizer backends:

    - **GFO (Gradient-Free-Optimizers)**: Converts to numpy arrays. Continuous
      ranges are discretized (default 100 points).
    - **Optuna**: Converts to Optuna distributions with native suggest_* calls.
    - **sklearn**: Converts to scipy distributions for RandomizedSearchCV.

    Use ``to_backend(backend_name)`` to get the converted format.

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
        Includes both regular dimensions and flattened nested dimensions.
    conditions : list[Condition]
        List of conditions that control when parameters are active.
        Includes auto-generated conditions from nested spaces.
    constraints : list[Constraint]
        List of constraints that filter invalid parameter combinations.
    nested_spaces : dict[str, dict[Any, SearchSpace]]
        Nested search spaces for hierarchical parameter structures.
        Maps parent parameter name to dict of {class/callable: SearchSpace}.

    See Also
    --------
    add_condition : Add conditional activation for a parameter.
    add_constraint : Add constraint filtering invalid combinations.
    union : Merge two search spaces.
    to_backend : Convert to backend-specific format.
    wrap_params : Wrap flat params for nested access.

    Examples
    --------
    **Basic usage with type inference:**

    >>> import numpy as np
    >>> space = SearchSpace(
    ...     x=np.arange(-10, 10, 0.1),       # discrete
    ...     lr=(1e-5, 1e-1, "log"),          # log-scale continuous
    ...     kernel=["rbf", "linear", "poly"], # categorical
    ...     seed=42,                          # constant
    ... )

    **Categorical with classes and functions:**

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> space = SearchSpace(
    ...     estimator=[RandomForestClassifier, SVC],  # list = categorical
    ...     activation=[torch.relu, torch.tanh],      # functions work too
    ... )

    **Conditions for conditional parameters (SVM example):**

    >>> space = SearchSpace(
    ...     kernel=["rbf", "linear", "poly"],
    ...     C=(0.01, 100.0, "log"),
    ...     gamma=(1e-4, 10.0, "log"),
    ...     degree=[2, 3, 4, 5],
    ... )
    >>> # gamma is irrelevant for linear kernel
    >>> space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")
    >>> # degree only applies to poly kernel
    >>> space.add_condition("degree", when=lambda p: p["kernel"] == "poly")

    **Constraints for parameter interactions:**

    >>> space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
    >>> space.add_constraint(lambda p: p["x"] + p["y"] < 15)

    **Union to combine spaces:**

    >>> base = SearchSpace(lr=(1e-5, 1e-1, "log"))
    >>> regularization = SearchSpace(dropout=(0.0, 0.5))
    >>> combined = base | regularization

    **Nested search spaces (model selection):**

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

    This automatically creates:

    - ``estimator``: categorical dimension [RandomForestClassifier, SVC]
    - ``randomforestclassifier__n_estimators``: active when estimator=RFC
    - ``randomforestclassifier__max_depth``: active when estimator=RFC
    - ``svc__C``: active when estimator=SVC
    - ``svc__kernel``: active when estimator=SVC

    **Accessing nested parameters in objective function:**

    >>> def objective(params):
    ...     wrapped = space.wrap_params(params)
    ...     # Access nested params transparently
    ...     n_est = wrapped["estimator"]["n_estimators"]
    ...     # Or instantiate directly
    ...     model = wrapped["estimator"]()
    ...     return score
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
        # Initialize components
        self._dim_registry = DimensionRegistry()
        self._cond_manager = ConditionManager(self._dim_registry)
        self._const_manager = ConstraintManager()
        self._nested_handler = NestedSpaceHandler(
            self._dim_registry, self._cond_manager
        )

        # Process dict-based space
        if __dict_space is not None:
            for name, value in __dict_space.items():
                self._dim_registry.add(name, value)

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
                if NestedSpaceHandler.looks_like_nested_space(value):
                    self._nested_handler.add(name, value)
                else:
                    self._dim_registry.add(name, value)
            else:
                self._dim_registry.add(name, value)

    # =========================================================================
    # Compatibility Properties - provide direct access to internal data
    # =========================================================================

    @property
    def dimensions(self) -> dict[str, Dimension]:
        """Dictionary mapping parameter names to Dimension objects."""
        return self._dim_registry.dimensions

    @property
    def conditions(self) -> list[Condition]:
        """List of conditions that control when parameters are active."""
        return self._cond_manager.conditions

    @property
    def constraints(self) -> list[Constraint]:
        """List of constraints that filter invalid parameter combinations."""
        return self._const_manager.constraints

    @property
    def nested_spaces(self) -> dict[str, dict[Any, "SearchSpace"]]:
        """Nested search spaces for hierarchical parameter structures."""
        return self._nested_handler.nested_spaces

    # =========================================================================
    # Condition Management
    # =========================================================================

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
        self._cond_manager.add(param, when, depends_on, name)
        return self

    # =========================================================================
    # Constraint Management
    # =========================================================================

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
        self._const_manager.add(constraint, name, params)
        return self

    # =========================================================================
    # Union Operations
    # =========================================================================

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
                result._dim_registry.add_dimension(dim)

        # Add/merge dimensions from other
        for name, dim in other.dimensions.items():
            # Skip dimensions from nested spaces that should be excluded
            if should_exclude_other(name):
                continue

            if name in result._dim_registry:
                existing_dim = result._dim_registry[name]

                # Check for type change - raise by default
                if not allow_type_change and existing_dim.dim_type != dim.dim_type:
                    raise ValueError(
                        f"Parameter '{name}' has conflicting types: "
                        f"{existing_dim.dim_type.value} vs {dim.dim_type.value}. "
                        f"Use allow_type_change=True to override."
                    )

                if on_conflict == "last":
                    result._dim_registry.add_dimension(dim)
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
                result._dim_registry.add_dimension(dim)

        # Merge conditions, excluding those targeting excluded dimensions
        result._cond_manager._conditions = [
            c for c in self.conditions if not should_exclude_self(c.target_param)
        ] + [c for c in other.conditions if not should_exclude_other(c.target_param)]

        # Merge constraints (constraints don't have target_param, keep all)
        result._const_manager._constraints = (
            self.constraints.copy() + other.constraints.copy()
        )

        # Merge nested spaces according to on_conflict strategy
        if on_conflict == "last":
            result._nested_handler._nested_spaces = {
                **self.nested_spaces,
                **other.nested_spaces,
            }
        elif on_conflict == "first":
            result._nested_handler._nested_spaces = {
                **other.nested_spaces,
                **self.nested_spaces,
            }
        else:  # "error" - conflicts already raised above
            result._nested_handler._nested_spaces = {
                **self.nested_spaces,
                **other.nested_spaces,
            }

        return result

    # =========================================================================
    # Query Properties
    # =========================================================================

    @property
    def param_names(self) -> list[str]:
        """Get list of all parameter names.

        Returns
        -------
        list[str]
            List of parameter names.
        """
        return self._dim_registry.param_names()

    @property
    def has_conditions(self) -> bool:
        """Check if the search space has any conditions.

        Returns
        -------
        bool
            True if there are conditions.
        """
        return self._cond_manager.has_conditions

    @property
    def has_constraints(self) -> bool:
        """Check if the search space has any constraints.

        Returns
        -------
        bool
            True if there are constraints.
        """
        return self._const_manager.has_constraints

    @property
    def has_nested_spaces(self) -> bool:
        """Check if the search space has nested spaces.

        Returns
        -------
        bool
            True if there are nested spaces.
        """
        return self._nested_handler.has_nested_spaces

    def get_dimension_types(self) -> dict[str, DimensionType]:
        """Get a mapping of parameter names to their dimension types.

        Returns
        -------
        dict[str, DimensionType]
            Dictionary mapping parameter names to dimension types.
        """
        return self._dim_registry.get_dimension_types()

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
        return self._dim_registry.has_dimension_type(dim_type)

    # =========================================================================
    # Runtime Evaluation
    # =========================================================================

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
        return self._cond_manager.filter_active(params)

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
        return self._const_manager.check_all(params)

    # =========================================================================
    # Backend Conversion
    # =========================================================================

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

    # =========================================================================
    # Parameter Wrapping
    # =========================================================================

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
            NestedSpaceConfig(parent_name=name)
            for name in self._nested_handler.nested_spaces.keys()
        )

        return ParamsView(
            flat_params=flat_params,
            nested_configs=configs,
            prefix_maker=make_prefix,
        )

    # =========================================================================
    # Container Protocol
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of dimensions."""
        return len(self._dim_registry)

    def __contains__(self, name: str) -> bool:
        """Check if a parameter name is in the search space."""
        return name in self._dim_registry

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self._dim_registry)

    def __repr__(self) -> str:
        """Return string representation."""
        dims = ", ".join(
            f"{k}={v.dim_type.value}" for k, v in self._dim_registry.items()
        )
        extras = []
        if self._cond_manager.has_conditions:
            extras.append(f"{len(self._cond_manager)} conditions")
        if self._const_manager.has_constraints:
            extras.append(f"{len(self._const_manager)} constraints")
        if self._nested_handler.has_nested_spaces:
            extras.append(f"{len(self._nested_handler)} nested spaces")

        extra_str = f" [{', '.join(extras)}]" if extras else ""
        return f"SearchSpace({dims}){extra_str}"
