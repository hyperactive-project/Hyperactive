"""Tests for search space feature extraction."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest

from hyperactive.algorithm_selection.ast_feature_engineering.search_space_features import (
    SearchSpaceFeatureExtractor,
    SearchSpaceFeatures,
    extract_search_space_features,
)


# =============================================================================
# Tests for SearchSpaceFeatures dataclass
# =============================================================================


class TestSearchSpaceFeaturesDataclass:
    """Test the SearchSpaceFeatures dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        features = SearchSpaceFeatures()

        assert features.n_dimensions == 0
        assert features.n_continuous == 0
        assert features.total_size == 0

    def test_to_dict(self):
        """to_dict should return a dictionary."""
        features = SearchSpaceFeatures(n_dimensions=3, n_continuous=2)
        d = features.to_dict()

        assert isinstance(d, dict)
        assert d["n_dimensions"] == 3
        assert d["n_continuous"] == 2

    def test_to_vector(self):
        """to_vector should return a list."""
        features = SearchSpaceFeatures(n_dimensions=3)
        v = features.to_vector()

        assert isinstance(v, list)
        assert 3 in v

    def test_feature_names(self):
        """feature_names should match to_dict keys."""
        names = SearchSpaceFeatures.feature_names()
        d = SearchSpaceFeatures().to_dict()

        assert names == list(d.keys())

    def test_has_mixed_types_converted_to_int(self):
        """has_mixed_types should be converted to int in to_dict."""
        features = SearchSpaceFeatures(has_mixed_types=True)
        d = features.to_dict()

        assert d["has_mixed_types"] == 1

        features2 = SearchSpaceFeatures(has_mixed_types=False)
        d2 = features2.to_dict()

        assert d2["has_mixed_types"] == 0


# =============================================================================
# Tests for basic dimension counting
# =============================================================================


class TestDimensionCounting:
    """Test counting of dimensions."""

    def test_empty_search_space(self):
        """Empty search space should have 0 dimensions."""
        search_space = {}

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 0

    def test_single_dimension(self):
        """Single dimension should be counted."""
        search_space = {"x": [1, 2, 3]}

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 1

    def test_multiple_dimensions(self):
        """Multiple dimensions should be counted."""
        search_space = {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 3


# =============================================================================
# Tests for parameter type detection
# =============================================================================


class TestParameterTypeDetection:
    """Test detection of parameter types."""

    def test_continuous_parameters(self):
        """Float parameters should be detected as continuous."""
        search_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "momentum": [0.9, 0.95, 0.99],
        }

        features = extract_search_space_features(search_space)

        assert features.n_continuous == 2
        assert features.n_discrete == 0
        assert features.n_categorical == 0

    def test_discrete_parameters(self):
        """Integer parameters should be detected as discrete."""
        search_space = {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [3, 5, 7, 10],
        }

        features = extract_search_space_features(search_space)

        assert features.n_discrete == 2
        assert features.n_continuous == 0
        assert features.n_categorical == 0

    def test_categorical_parameters(self):
        """String parameters should be detected as categorical."""
        search_space = {
            "criterion": ["gini", "entropy"],
            "kernel": ["linear", "rbf", "poly"],
        }

        features = extract_search_space_features(search_space)

        assert features.n_categorical == 2
        assert features.n_continuous == 0
        assert features.n_discrete == 0

    def test_boolean_parameters(self):
        """Boolean parameters should be detected as categorical."""
        search_space = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }

        features = extract_search_space_features(search_space)

        assert features.n_categorical == 2

    def test_mixed_parameters(self):
        """Mixed parameter types should be counted correctly."""
        search_space = {
            "learning_rate": [0.01, 0.1],  # continuous
            "n_estimators": [10, 100],  # discrete
            "criterion": ["gini", "entropy"],  # categorical
        }

        features = extract_search_space_features(search_space)

        assert features.n_continuous == 1
        assert features.n_discrete == 1
        assert features.n_categorical == 1
        assert features.has_mixed_types is True


# =============================================================================
# Tests for search space size calculations
# =============================================================================


class TestSearchSpaceSize:
    """Test calculation of search space size."""

    def test_total_size_calculation(self):
        """Total size should be product of all choices."""
        search_space = {
            "x": [1, 2, 3],  # 3 choices
            "y": [4, 5],  # 2 choices
        }

        features = extract_search_space_features(search_space)

        assert features.total_size == 6  # 3 * 2

    def test_large_search_space_capped(self):
        """Very large search spaces should be capped."""
        search_space = {f"dim_{i}": list(range(100)) for i in range(20)}

        features = extract_search_space_features(search_space)

        # Should be capped at 10^15
        assert features.total_size <= 10**15

    def test_avg_choices_per_dim(self):
        """Average choices per dimension should be calculated."""
        search_space = {
            "x": [1, 2, 3, 4],  # 4 choices
            "y": [5, 6],  # 2 choices
        }

        features = extract_search_space_features(search_space)

        assert features.avg_choices_per_dim == 3.0  # (4 + 2) / 2

    def test_min_max_choices(self):
        """Min and max choices should be tracked."""
        search_space = {
            "x": [1, 2],  # 2 choices
            "y": [3, 4, 5, 6, 7],  # 5 choices
            "z": [8, 9, 10],  # 3 choices
        }

        features = extract_search_space_features(search_space)

        assert features.min_choices == 2
        assert features.max_choices == 5


# =============================================================================
# Tests for range span calculation
# =============================================================================


class TestRangeSpan:
    """Test calculation of average range span."""

    def test_range_span_numeric(self):
        """Range span should be calculated for numeric parameters."""
        search_space = {
            "x": [0.0, 1.0],  # range = 1, span = 1/1 = 1
        }

        features = extract_search_space_features(search_space)

        assert features.avg_range_span > 0

    def test_range_span_ignores_categorical(self):
        """Range span should not include categorical parameters."""
        search_space = {
            "x": [0.0, 10.0],  # numeric
            "y": ["a", "b", "c"],  # categorical
        }

        features = extract_search_space_features(search_space)

        # Should only consider the numeric parameter
        assert features.avg_range_span > 0


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_value_list(self):
        """Empty value list should be handled."""
        search_space = {"x": []}

        features = extract_search_space_features(search_space)

        # Should handle gracefully
        assert features.n_dimensions == 1

    def test_single_value_parameter(self):
        """Single value parameter should be handled."""
        search_space = {"x": [5]}

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 1
        assert features.min_choices == 1
        assert features.max_choices == 1

    def test_none_in_search_space(self):
        """None values should be handled as categorical."""
        search_space = {"x": [None, 1, 2]}

        features = extract_search_space_features(search_space)

        # Mixed None and int should be categorical
        assert features.n_categorical == 1

    def test_tuple_values(self):
        """Tuple values should be handled."""
        search_space = {"x": (1, 2, 3)}  # tuple instead of list

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 1

    def test_mixed_int_float(self):
        """Mixed int and float should be continuous."""
        search_space = {"x": [1, 2.5, 3]}

        features = extract_search_space_features(search_space)

        assert features.n_continuous == 1


# =============================================================================
# Tests for has_mixed_types
# =============================================================================


class TestHasMixedTypes:
    """Test the has_mixed_types flag."""

    def test_all_same_type_not_mixed(self):
        """All same type should not be mixed."""
        search_space = {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }

        features = extract_search_space_features(search_space)

        assert features.has_mixed_types is False

    def test_different_types_is_mixed(self):
        """Different types should be mixed."""
        search_space = {
            "x": [1, 2, 3],  # discrete
            "y": ["a", "b"],  # categorical
        }

        features = extract_search_space_features(search_space)

        assert features.has_mixed_types is True

    def test_continuous_and_discrete_is_mixed(self):
        """Continuous and discrete should be mixed."""
        search_space = {
            "x": [0.1, 0.2, 0.3],  # continuous
            "y": [1, 2, 3],  # discrete
        }

        features = extract_search_space_features(search_space)

        assert features.has_mixed_types is True


# =============================================================================
# Tests for convenience function
# =============================================================================


class TestConvenienceFunction:
    """Test the extract_search_space_features convenience function."""

    def test_basic_usage(self):
        """Basic usage should work."""
        search_space = {"x": [1, 2, 3], "y": [4, 5, 6]}

        features = extract_search_space_features(search_space)

        assert isinstance(features, SearchSpaceFeatures)
        assert features.n_dimensions == 2

    def test_returns_same_as_extractor(self):
        """Should return same result as using extractor directly."""
        search_space = {"x": [1, 2, 3]}

        features1 = extract_search_space_features(search_space)

        extractor = SearchSpaceFeatureExtractor()
        features2 = extractor.extract(search_space)

        assert features1.to_dict() == features2.to_dict()


# =============================================================================
# Tests for realistic search spaces
# =============================================================================


class TestRealisticSearchSpaces:
    """Test with realistic hyperparameter search spaces."""

    def test_sklearn_random_forest_space(self):
        """Test Random Forest hyperparameter space."""
        search_space = {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 4
        assert features.has_mixed_types is True

    def test_neural_network_space(self):
        """Test neural network hyperparameter space."""
        search_space = {
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64, 128],
            "hidden_units": [32, 64, 128, 256],
            "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
            "activation": ["relu", "tanh", "sigmoid"],
        }

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 5
        assert features.n_categorical >= 1
        assert features.total_size == 4 * 4 * 4 * 5 * 3

    def test_xgboost_space(self):
        """Test XGBoost hyperparameter space."""
        search_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        }

        features = extract_search_space_features(search_space)

        assert features.n_dimensions == 5
        assert features.n_discrete >= 1
        assert features.n_continuous >= 1
