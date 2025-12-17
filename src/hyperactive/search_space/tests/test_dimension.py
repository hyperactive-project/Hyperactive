"""Tests for dimension type inference."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.search_space._dimension import DimensionType, infer_dimension


class TestDimensionTypeInference:
    """Test automatic dimension type inference from Python types."""

    def test_list_infers_categorical(self):
        """List values should infer categorical dimension."""
        dim = infer_dimension("x", [1, 2, 3])
        assert dim.dim_type == DimensionType.CATEGORICAL
        assert dim.values == [1, 2, 3]

    def test_list_with_strings(self):
        """List of strings should infer categorical with str dtype."""
        dim = infer_dimension("kernel", ["rbf", "linear", "poly"])
        assert dim.dim_type == DimensionType.CATEGORICAL
        assert dim.dtype == str

    def test_list_with_classes(self):
        """Classes should work as categorical values."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        dim = infer_dimension("est", [RandomForestClassifier, SVC])
        assert dim.dim_type == DimensionType.CATEGORICAL
        assert RandomForestClassifier in dim.values
        assert SVC in dim.values

    def test_tuple_floats_infers_continuous(self):
        """Float tuple should infer continuous float dimension."""
        dim = infer_dimension("x", (0.0, 10.0))
        assert dim.dim_type == DimensionType.CONTINUOUS
        assert dim.dtype == float
        assert dim.low == 0.0
        assert dim.high == 10.0

    def test_tuple_ints_infers_continuous_int(self):
        """Integer tuple should infer continuous integer dimension."""
        dim = infer_dimension("n", (1, 100))
        assert dim.dim_type == DimensionType.CONTINUOUS
        assert dim.dtype == int
        assert dim.low == 1
        assert dim.high == 100

    def test_tuple_with_log_infers_log_scale(self):
        """Tuple with 'log' marker should infer log-scale dimension."""
        dim = infer_dimension("lr", (1e-5, 1e-1, "log"))
        assert dim.dim_type == DimensionType.CONTINUOUS_LOG
        assert dim.log_scale is True
        assert dim.low == 1e-5
        assert dim.high == 1e-1

    def test_numpy_array_infers_discrete(self):
        """Numpy array should infer discrete dimension."""
        arr = np.arange(0, 10, 0.5)
        dim = infer_dimension("x", arr)
        assert dim.dim_type == DimensionType.DISCRETE
        assert len(dim.values) == 20

    def test_numpy_int_array(self):
        """Numpy integer array should have int dtype."""
        arr = np.arange(10)
        dim = infer_dimension("x", arr)
        assert dim.dim_type == DimensionType.DISCRETE
        assert dim.dtype == int

    def test_scipy_distribution_detected(self):
        """Scipy distributions should be detected."""
        import scipy.stats as st

        dim = infer_dimension("x", st.uniform(0, 1))
        assert dim.dim_type == DimensionType.DISTRIBUTION

    def test_scalar_int_infers_constant(self):
        """Scalar int should infer constant dimension."""
        dim = infer_dimension("seed", 42)
        assert dim.dim_type == DimensionType.CONSTANT
        assert dim.values == 42

    def test_scalar_str_infers_constant(self):
        """Scalar string should infer constant dimension."""
        dim = infer_dimension("name", "model_v1")
        assert dim.dim_type == DimensionType.CONSTANT

    def test_none_in_list_works(self):
        """None should be allowed in categorical lists."""
        dim = infer_dimension("max_depth", [3, 5, 10, None])
        assert dim.dim_type == DimensionType.CATEGORICAL
        assert None in dim.values

    def test_invalid_type_raises(self):
        """Invalid types should raise TypeError."""
        with pytest.raises(TypeError, match="Cannot infer dimension type"):
            infer_dimension("x", {"a": 1})  # dict not supported

    def test_invalid_tuple_raises(self):
        """Invalid tuple format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tuple specification"):
            infer_dimension("x", (1, 2, 3, 4))  # too many elements

    def test_log_scale_with_negative_raises(self):
        """Log-scale with non-positive bounds should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            infer_dimension("x", (-1, 1, "log"))

    def test_empty_list_dtype(self):
        """Empty list should have object dtype."""
        dim = infer_dimension("x", [])
        assert dim.dtype == object


class TestDimensionRepresentation:
    """Test Dimension string representation."""

    def test_repr_categorical(self):
        """Test repr for categorical dimension."""
        dim = infer_dimension("x", [1, 2, 3])
        assert "categorical" in repr(dim)

    def test_repr_log_scale(self):
        """Test repr for log-scale dimension."""
        dim = infer_dimension("lr", (1e-5, 1e-1, "log"))
        assert "log" in repr(dim)
