"""Search space adapter for optimizer backends."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

__all__ = ["SearchSpaceAdapter"]

# Default number of points for discretizing continuous dimensions
DEFAULT_N_POINTS = 100


class SearchSpaceAdapter:
    """Adapts search spaces for optimizer backends.

    Handles:
    - Encoding/decoding of categorical dimensions (strings to integers)
    - Discretization of continuous dimensions (tuples to lists)

    Parameters
    ----------
    search_space : dict[str, list | tuple]
        The search space as a dictionary mapping parameter names to:
        - list: discrete values (categorical if contains strings, numeric otherwise)
        - tuple: continuous range, formats supported:
            - (low, high) - linear scale, 100 points
            - (low, high, "log") - log scale, 100 points
            - (low, high, n_points) - linear scale, n_points
            - (low, high, n_points, "log") - log scale, n_points
    capabilities : dict
        Backend capability tags, e.g., {"categorical": True, "continuous": False}.

    Attributes
    ----------
    needs_adaptation : bool
        Whether any dimensions require encoding or discretization.
    categorical_mapping : dict
        Mapping of {param_name: {index: original_value}} for encoded dimensions.

    Examples
    --------
    >>> space = {"kernel": ["rbf", "linear"], "C": (0.01, 10.0)}
    >>> adapter = SearchSpaceAdapter(space, {"categorical": False, "continuous": False})
    >>> adapter.validate()  # Raises if invalid
    >>> encoded = adapter.encode()
    >>> encoded
    {"kernel": [0, 1], "C": [0.01, 0.11, 0.21, ...]}  # 100 points
    >>> adapter.decode({"kernel": 1, "C": 0.5})
    {"kernel": "linear", "C": 0.5}
    """

    def __init__(self, search_space: dict, capabilities: dict):
        self._original_space = search_space
        self._capabilities = capabilities
        self._categorical_mapping = {}
        self._continuous_dims = set()

    @property
    def needs_adaptation(self) -> bool:
        """Whether any adaptation (encoding or discretization) is needed."""
        return self._needs_categorical_encoding() or self._needs_discretization()

    # Keep old property for backwards compatibility
    @property
    def needs_encoding(self) -> bool:
        """Whether encoding is needed (deprecated, use needs_adaptation)."""
        return self.needs_adaptation

    @property
    def categorical_mapping(self) -> dict:
        """Mapping of encoded categorical dimensions."""
        return self._categorical_mapping

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self):
        """Validate the search space format.

        Raises
        ------
        ValueError
            If the search space contains invalid definitions.
        TypeError
            If values have unexpected types.
        """
        if not isinstance(self._original_space, dict):
            raise TypeError(
                f"Search space must be a dict, got {type(self._original_space).__name__}"
            )

        if not self._original_space:
            raise ValueError("Search space cannot be empty")

        for name, values in self._original_space.items():
            self._validate_dimension(name, values)

    def _validate_dimension(self, name: str, values):
        """Validate a single dimension."""
        if isinstance(values, tuple):
            self._validate_continuous(name, values)
        elif isinstance(values, (list, np.ndarray)):
            self._validate_discrete(name, values)
        else:
            raise TypeError(
                f"Parameter '{name}': expected list (discrete) or tuple (continuous), "
                f"got {type(values).__name__}. "
                f"Use [a, b, c] for discrete values or (low, high) for continuous ranges."
            )

    def _validate_continuous(self, name: str, values: tuple):
        """Validate a continuous dimension (tuple)."""
        if len(values) < 2:
            raise ValueError(
                f"Parameter '{name}': continuous range needs at least 2 values "
                f"(low, high), got {len(values)}. "
                f"Example: (0.01, 10.0) or (1e-5, 1e-1, 'log')"
            )

        if len(values) > 4:
            raise ValueError(
                f"Parameter '{name}': continuous range has too many values "
                f"({len(values)}). Supported formats:\n"
                f"  (low, high)              - linear, 100 points\n"
                f"  (low, high, 'log')       - log scale, 100 points\n"
                f"  (low, high, n_points)    - linear, n_points\n"
                f"  (low, high, n_points, 'log') - log scale, n_points"
            )

        low, high = values[0], values[1]

        # Check low and high are numeric
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError(
                f"Parameter '{name}': low and high must be numeric, "
                f"got low={type(low).__name__}, high={type(high).__name__}"
            )

        # Check low < high
        if low >= high:
            raise ValueError(
                f"Parameter '{name}': low ({low}) must be less than high ({high})"
            )

        # Parse and validate optional arguments
        n_points, log_scale = self._parse_continuous_options(name, values)

        # Check log scale with non-positive values
        if log_scale and low <= 0:
            raise ValueError(
                f"Parameter '{name}': log scale requires positive values, "
                f"but low={low}. Use linear scale or adjust the range."
            )

        # Check n_points
        if n_points < 2:
            raise ValueError(
                f"Parameter '{name}': n_points must be at least 2, got {n_points}"
            )

    def _validate_discrete(self, name: str, values):
        """Validate a discrete dimension (list)."""
        if len(values) == 0:
            raise ValueError(
                f"Parameter '{name}': discrete list cannot be empty. "
                f"Provide at least one value."
            )

    def _parse_continuous_options(self, name: str, values: tuple):
        """Parse optional n_points and log_scale from tuple.

        Returns
        -------
        n_points : int
        log_scale : bool
        """
        n_points = DEFAULT_N_POINTS
        log_scale = False

        if len(values) == 2:
            # (low, high)
            pass
        elif len(values) == 3:
            # (low, high, "log") or (low, high, n_points)
            third = values[2]
            if isinstance(third, str):
                if third.lower() != "log":
                    raise ValueError(
                        f"Parameter '{name}': unknown scale '{third}'. "
                        f"Use 'log' for logarithmic scale."
                    )
                log_scale = True
            elif isinstance(third, (int, float)):
                n_points = int(third)
            else:
                raise TypeError(
                    f"Parameter '{name}': third value must be 'log' or n_points (int), "
                    f"got {type(third).__name__}"
                )
        elif len(values) == 4:
            # (low, high, n_points, "log")
            third, fourth = values[2], values[3]
            if not isinstance(third, (int, float)):
                raise TypeError(
                    f"Parameter '{name}': n_points must be numeric, "
                    f"got {type(third).__name__}"
                )
            n_points = int(third)
            if not isinstance(fourth, str) or fourth.lower() != "log":
                raise ValueError(
                    f"Parameter '{name}': fourth value must be 'log', "
                    f"got '{fourth}'"
                )
            log_scale = True

        return n_points, log_scale

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------

    def _needs_categorical_encoding(self) -> bool:
        """Check if categorical encoding is needed."""
        if self._capabilities.get("categorical", True):
            return False
        return self._has_categorical_values()

    def _needs_discretization(self) -> bool:
        """Check if continuous discretization is needed."""
        if self._capabilities.get("continuous", False):
            return False
        return self._has_continuous_values()

    def _has_categorical_values(self) -> bool:
        """Detect if any dimension contains string values."""
        for values in self._original_space.values():
            if self._is_categorical(values):
                return True
        return False

    def _has_continuous_values(self) -> bool:
        """Detect if any dimension is continuous (tuple)."""
        for values in self._original_space.values():
            if self._is_continuous(values):
                return True
        return False

    def _is_categorical(self, values) -> bool:
        """Check if a dimension's values are categorical (list containing strings)."""
        if isinstance(values, tuple):
            return False  # Tuples are continuous, not categorical
        if hasattr(values, "__iter__") and not isinstance(values, str):
            return any(isinstance(v, str) for v in values)
        return False

    def _is_continuous(self, values) -> bool:
        """Check if a dimension is continuous (tuple)."""
        return isinstance(values, tuple)

    # -------------------------------------------------------------------------
    # Encoding / Discretization
    # -------------------------------------------------------------------------

    def encode(self) -> dict:
        """Encode the search space for the backend.

        - Categorical dimensions (lists with strings) are converted to integers
        - Continuous dimensions (tuples) are discretized to lists

        Returns
        -------
        dict
            Encoded search space ready for the backend.
        """
        if not self.needs_adaptation:
            return self._original_space

        self._categorical_mapping = {}
        self._continuous_dims = set()
        encoded = {}

        needs_cat_encoding = self._needs_categorical_encoding()
        needs_discretize = self._needs_discretization()

        for name, values in self._original_space.items():
            if self._is_continuous(values) and needs_discretize:
                # Discretize continuous dimension
                encoded[name] = self._discretize(name, values)
                self._continuous_dims.add(name)
            elif self._is_categorical(values) and needs_cat_encoding:
                # Encode categorical dimension
                self._categorical_mapping[name] = {i: v for i, v in enumerate(values)}
                encoded[name] = list(range(len(values)))
            else:
                encoded[name] = values

        return encoded

    def _discretize(self, name: str, values: tuple) -> list:
        """Convert continuous range to discrete list of values.

        Parameters
        ----------
        name : str
            Parameter name (for error messages).
        values : tuple
            Continuous range specification.

        Returns
        -------
        list
            Discretized values.
        """
        low, high = values[0], values[1]
        n_points, log_scale = self._parse_continuous_options(name, values)

        if log_scale:
            return np.geomspace(low, high, n_points).tolist()
        else:
            return np.linspace(low, high, n_points).tolist()

    def decode(self, params: dict) -> dict:
        """Decode backend results to original format.

        Integer indices for categorical dimensions are converted back
        to their original string values.

        Note: Continuous dimensions don't need decoding - the discretized
        float values are already valid.

        Parameters
        ----------
        params : dict
            Parameter dictionary from the optimizer.

        Returns
        -------
        dict
            Parameters with original categorical values restored.
        """
        if not self._categorical_mapping:
            return params

        decoded = params.copy()
        for name, mapping in self._categorical_mapping.items():
            if name in decoded:
                val = decoded[name]
                # Handle numpy types
                if hasattr(val, "item"):
                    val = val.item()
                decoded[name] = mapping[int(val)]

        return decoded

    def wrap_experiment(self, experiment):
        """Wrap experiment to decode params before scoring.

        During optimization, the backend calls experiment.score(params)
        with encoded values. This wrapper decodes them first.

        Parameters
        ----------
        experiment : BaseExperiment
            The original experiment.

        Returns
        -------
        experiment
            Wrapped experiment that decodes params, or original if no encoding.
        """
        if not self._categorical_mapping:
            return experiment

        return _DecodingExperimentWrapper(experiment, self._categorical_mapping)


class _DecodingExperimentWrapper:
    """Wrapper that decodes params before passing to experiment."""

    def __init__(self, experiment, categorical_mapping):
        self._experiment = experiment
        self._mapping = categorical_mapping

    def _decode(self, params):
        decoded = params.copy()
        for name, mapping in self._mapping.items():
            if name in decoded:
                val = decoded[name]
                if hasattr(val, "item"):
                    val = val.item()
                decoded[name] = mapping[int(val)]
        return decoded

    def score(self, params):
        return self._experiment.score(self._decode(params))

    def __call__(self, params):
        return self.score(params)

    def evaluate(self, params):
        return self._experiment.evaluate(self._decode(params))

    def __getattr__(self, name):
        # Forward all other attributes to wrapped experiment
        return getattr(self._experiment, name)
