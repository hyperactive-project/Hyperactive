"""Search space adapter for optimizer backends."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

__all__ = ["SearchSpaceAdapter"]


class SearchSpaceAdapter:
    """Adapts search spaces for optimizer backends.

    Handles encoding/decoding of categorical dimensions when the backend
    doesn't support them natively.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space as a dictionary mapping parameter names to value lists.
    capabilities : dict
        Backend capability tags, e.g., {"categorical": True, "continuous": False}.

    Attributes
    ----------
    needs_encoding : bool
        Whether any dimensions require encoding.
    categorical_mapping : dict
        Mapping of {param_name: {index: original_value}} for encoded dimensions.

    Examples
    --------
    >>> space = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10]}
    >>> adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
    >>> encoded = adapter.encode()
    >>> encoded
    {"kernel": [0, 1], "C": [0.1, 1, 10]}
    >>> adapter.decode({"kernel": 1, "C": 1})
    {"kernel": "linear", "C": 1}
    """

    def __init__(self, search_space: dict, capabilities: dict):
        self._original_space = search_space
        self._capabilities = capabilities
        self._categorical_mapping = {}
        self._needs_encoding = self._check_needs_encoding()

    @property
    def needs_encoding(self) -> bool:
        """Whether encoding is needed."""
        return self._needs_encoding

    @property
    def categorical_mapping(self) -> dict:
        """Mapping of encoded categorical dimensions."""
        return self._categorical_mapping

    def _check_needs_encoding(self) -> bool:
        """Check if encoding is needed based on capabilities and search space."""
        # If backend supports categorical natively, no encoding needed
        if self._capabilities.get("categorical", True):
            return False
        # Check if search space contains categorical values
        return self._has_categorical_values()

    def _has_categorical_values(self) -> bool:
        """Detect if any dimension contains string values."""
        for values in self._original_space.values():
            if self._is_categorical(values):
                return True
        return False

    def _is_categorical(self, values) -> bool:
        """Check if a dimension's values are categorical (contain strings)."""
        if hasattr(values, "__iter__") and not isinstance(values, str):
            return any(isinstance(v, str) for v in values)
        return False

    def encode(self) -> dict:
        """Encode the search space for the backend.

        Categorical dimensions (containing strings) are converted to
        integer indices. The mapping is stored for later decoding.

        Returns
        -------
        dict
            Encoded search space with categorical values as integers.
        """
        if not self._needs_encoding:
            return self._original_space

        self._categorical_mapping = {}
        encoded = {}

        for name, values in self._original_space.items():
            if self._is_categorical(values):
                # Store mapping: {index: original_value}
                self._categorical_mapping[name] = {i: v for i, v in enumerate(values)}
                # Replace with integer indices
                encoded[name] = list(range(len(values)))
            else:
                encoded[name] = values

        return encoded

    def decode(self, params: dict) -> dict:
        """Decode backend results to original format.

        Integer indices for categorical dimensions are converted back
        to their original string values.

        Parameters
        ----------
        params : dict
            Parameter dictionary from the optimizer, potentially with
            encoded categorical values.

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
