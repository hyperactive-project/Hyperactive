"""Search space feature extraction for algorithm selection.

This module extracts features from optimization search spaces that may
be predictive of which algorithm will perform best.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchSpaceFeatures:
    """Container for search space features.

    Attributes
    ----------
    n_dimensions : int
        Number of parameters in the search space.
    n_continuous : int
        Number of continuous (float) parameters.
    n_discrete : int
        Number of discrete (int) parameters.
    n_categorical : int
        Number of categorical parameters.
    total_size : int
        Total number of possible combinations (may be very large or infinite).
    avg_choices_per_dim : float
        Average number of choices per dimension.
    min_choices : int
        Minimum number of choices across dimensions.
    max_choices : int
        Maximum number of choices across dimensions.
    has_mixed_types : bool
        Whether the search space has mixed parameter types.
    avg_range_span : float
        Average normalized range span for numeric parameters.
    """

    n_dimensions: int = 0
    n_continuous: int = 0
    n_discrete: int = 0
    n_categorical: int = 0
    total_size: int = 0
    avg_choices_per_dim: float = 0.0
    min_choices: int = 0
    max_choices: int = 0
    has_mixed_types: bool = False
    avg_range_span: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert features to dictionary.

        Returns
        -------
        dict
            Dictionary with feature names as keys.
        """
        return {
            "n_dimensions": self.n_dimensions,
            "n_continuous": self.n_continuous,
            "n_discrete": self.n_discrete,
            "n_categorical": self.n_categorical,
            "total_size": self.total_size,
            "avg_choices_per_dim": self.avg_choices_per_dim,
            "min_choices": self.min_choices,
            "max_choices": self.max_choices,
            "has_mixed_types": int(self.has_mixed_types),
            "avg_range_span": self.avg_range_span,
        }

    def to_vector(self) -> list[float]:
        """Convert features to a list (for ML models).

        Returns
        -------
        list of float
            Feature values in consistent order.
        """
        return list(self.to_dict().values())

    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names in consistent order.

        Returns
        -------
        list of str
            Feature names matching to_vector() order.
        """
        return list(SearchSpaceFeatures().to_dict().keys())


class SearchSpaceFeatureExtractor:
    """Extract features from optimization search spaces.

    The search space is expected to be a dictionary mapping parameter
    names to lists of possible values, following Hyperactive conventions.

    Examples
    --------
    >>> search_space = {
    ...     "learning_rate": [0.001, 0.01, 0.1],
    ...     "n_estimators": [10, 50, 100, 200],
    ...     "criterion": ["gini", "entropy"],
    ... }
    >>> extractor = SearchSpaceFeatureExtractor()
    >>> features = extractor.extract(search_space)
    >>> features.n_dimensions
    3
    """

    def extract(self, search_space: dict[str, list]) -> SearchSpaceFeatures:
        """Extract features from a search space.

        Parameters
        ----------
        search_space : dict
            Search space dictionary mapping parameter names to lists of values.

        Returns
        -------
        SearchSpaceFeatures
            Extracted features.
        """
        features = SearchSpaceFeatures()

        if not search_space:
            return features

        features.n_dimensions = len(search_space)

        choices_per_dim = []
        type_counts = {"continuous": 0, "discrete": 0, "categorical": 0}
        range_spans = []

        for name, values in search_space.items():
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                continue

            n_choices = len(values)
            choices_per_dim.append(n_choices)

            # Determine parameter type
            param_type = self._infer_param_type(values)
            type_counts[param_type] += 1

            # Calculate range span for numeric types
            if param_type in ("continuous", "discrete"):
                try:
                    numeric_values = [float(v) for v in values]
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    if max_val != min_val:
                        # Normalize by the range
                        span = (max_val - min_val) / max(abs(max_val), abs(min_val), 1)
                        range_spans.append(span)
                except (ValueError, TypeError):
                    pass

        # Populate features
        features.n_continuous = type_counts["continuous"]
        features.n_discrete = type_counts["discrete"]
        features.n_categorical = type_counts["categorical"]

        if choices_per_dim:
            features.avg_choices_per_dim = sum(choices_per_dim) / len(choices_per_dim)
            features.min_choices = min(choices_per_dim)
            features.max_choices = max(choices_per_dim)

            # Calculate total size (product of all choices)
            total = 1
            for n in choices_per_dim:
                total *= n
                # Cap at a large number to avoid overflow
                if total > 10**15:
                    total = 10**15
                    break
            features.total_size = total

        if range_spans:
            features.avg_range_span = sum(range_spans) / len(range_spans)

        # Check for mixed types
        active_types = sum(1 for count in type_counts.values() if count > 0)
        features.has_mixed_types = active_types > 1

        return features

    def _infer_param_type(self, values: list) -> str:
        """Infer the parameter type from its values.

        Parameters
        ----------
        values : list
            List of possible values.

        Returns
        -------
        str
            One of "continuous", "discrete", or "categorical".
        """
        if not values:
            return "categorical"

        # Check if all values are strings (categorical)
        if all(isinstance(v, str) for v in values):
            return "categorical"

        # Check if all values are booleans (categorical)
        if all(isinstance(v, bool) for v in values):
            return "categorical"

        # Check if all values are integers
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return "discrete"

        # Check if all values are numeric (float or int)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return "continuous"

        # Mixed or unknown - treat as categorical
        return "categorical"


def extract_search_space_features(search_space: dict[str, list]) -> SearchSpaceFeatures:
    """Convenience function to extract search space features.

    Parameters
    ----------
    search_space : dict
        Search space dictionary.

    Returns
    -------
    SearchSpaceFeatures
        Extracted features.

    Examples
    --------
    >>> search_space = {"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]}  # doctest: +SKIP
    >>> features = extract_search_space_features(search_space)  # doctest: +SKIP
    >>> features.n_dimensions  # doctest: +SKIP
    2
    """
    extractor = SearchSpaceFeatureExtractor()
    return extractor.extract(search_space)
