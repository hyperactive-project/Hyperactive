"""Tests for SearchSpace class."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.search_space import SearchSpace
from hyperactive.search_space._dimension import DimensionType


class TestSearchSpaceCreation:
    """Test SearchSpace instantiation patterns."""

    def test_keyword_arguments(self):
        """Test creating SearchSpace with keyword arguments."""
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 10.0))
        assert "x" in space.dimensions
        assert "y" in space.dimensions

    def test_dict_argument(self):
        """Test creating SearchSpace with dict argument."""
        space = SearchSpace({"x": [1, 2, 3], "y": (0.0, 10.0)})
        assert "x" in space.dimensions
        assert "y" in space.dimensions

    def test_mixed_dict_and_kwargs(self):
        """Test dict and kwargs can be combined."""
        space = SearchSpace({"x": [1, 2, 3]}, y=(0.0, 10.0))
        assert "x" in space.dimensions
        assert "y" in space.dimensions

    def test_param_names_property(self):
        """Test param_names returns all parameter names."""
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 10.0), z=42)
        assert set(space.param_names) == {"x", "y", "z"}

    def test_len(self):
        """Test __len__ returns number of dimensions."""
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 10.0))
        assert len(space) == 2

    def test_contains(self):
        """Test __contains__ checks parameter existence."""
        space = SearchSpace(x=[1, 2, 3])
        assert "x" in space
        assert "y" not in space

    def test_iter(self):
        """Test __iter__ iterates over parameter names."""
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 10.0))
        names = list(space)
        assert set(names) == {"x", "y"}


class TestSearchSpaceUnion:
    """Test union operation via | operator."""

    def test_union_combines_params(self):
        """Test union merges parameters from both spaces."""
        s1 = SearchSpace(x=[1, 2, 3])
        s2 = SearchSpace(y=(0.0, 10.0))
        combined = s1 | s2
        assert "x" in combined.dimensions
        assert "y" in combined.dimensions

    def test_union_last_wins_on_conflict(self):
        """Test conflict resolution: last wins by default."""
        s1 = SearchSpace(x=[1, 2, 3])
        s2 = SearchSpace(x=[4, 5, 6])
        combined = s1 | s2
        assert combined.dimensions["x"].values == [4, 5, 6]

    def test_union_first_wins(self):
        """Test conflict resolution: first wins option."""
        s1 = SearchSpace(x=[1, 2, 3])
        s2 = SearchSpace(x=[4, 5, 6])
        combined = s1.union(s2, on_conflict="first")
        assert combined.dimensions["x"].values == [1, 2, 3]

    def test_union_error_on_conflict(self):
        """Test conflict resolution: error option."""
        s1 = SearchSpace(x=[1, 2, 3])
        s2 = SearchSpace(x=[4, 5, 6])
        with pytest.raises(ValueError, match="Conflicting parameter"):
            s1.union(s2, on_conflict="error")

    def test_union_preserves_conditions(self):
        """Test union preserves conditions from both spaces."""
        s1 = SearchSpace(kernel=["rbf", "linear"], gamma=(0.1, 10.0))
        s1.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        s2 = SearchSpace(C=(0.01, 100.0))

        combined = s1 | s2
        assert len(combined.conditions) == 1

    def test_union_preserves_constraints(self):
        """Test union preserves constraints from both spaces."""
        s1 = SearchSpace(x=(0.0, 10.0))
        s1.add_constraint(lambda p: p["x"] < 5)
        s2 = SearchSpace(y=(0.0, 10.0))

        combined = s1 | s2
        assert len(combined.constraints) == 1


class TestSearchSpaceConditions:
    """Test conditional dimensions."""

    def test_add_condition_returns_self(self):
        """Test add_condition returns self for chaining."""
        space = SearchSpace(x=[1, 2], y=[3, 4])
        result = space.add_condition("y", when=lambda p: p["x"] == 1)
        assert result is space

    def test_add_condition_unknown_param_raises(self):
        """Test add_condition raises for unknown parameter."""
        space = SearchSpace(x=[1, 2])
        with pytest.raises(ValueError, match="Unknown parameter"):
            space.add_condition("unknown", when=lambda p: True)

    def test_condition_stored(self):
        """Test condition is stored correctly."""
        space = SearchSpace(x=[1, 2], y=[3, 4])
        space.add_condition("y", when=lambda p: p["x"] == 1)
        assert len(space.conditions) == 1
        assert space.conditions[0].target_param == "y"

    def test_has_conditions_property(self):
        """Test has_conditions property."""
        space = SearchSpace(x=[1, 2], y=[3, 4])
        assert not space.has_conditions
        space.add_condition("y", when=lambda p: p["x"] == 1)
        assert space.has_conditions


class TestSearchSpaceConstraints:
    """Test constraints."""

    def test_add_constraint_returns_self(self):
        """Test add_constraint returns self for chaining."""
        space = SearchSpace(x=(0.0, 10.0))
        result = space.add_constraint(lambda p: p["x"] < 5)
        assert result is space

    def test_constraint_stored(self):
        """Test constraint is stored correctly."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 10, name="sum_constraint")
        assert len(space.constraints) == 1
        assert space.constraints[0].name == "sum_constraint"

    def test_has_constraints_property(self):
        """Test has_constraints property."""
        space = SearchSpace(x=(0.0, 10.0))
        assert not space.has_constraints
        space.add_constraint(lambda p: p["x"] < 5)
        assert space.has_constraints

    def test_method_chaining(self):
        """Test method chaining for conditions and constraints."""
        space = (
            SearchSpace(x=[1, 2], y=[3, 4], z=(0.0, 10.0))
            .add_condition("y", when=lambda p: p["x"] == 1)
            .add_constraint(lambda p: p["z"] < 5)
        )
        assert len(space.conditions) == 1
        assert len(space.constraints) == 1

    def test_check_constraints(self):
        """Test check_constraints method."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 10)

        assert space.check_constraints({"x": 3, "y": 4})  # 7 < 10
        assert not space.check_constraints({"x": 5, "y": 6})  # 11 >= 10


class TestSearchSpaceNestedSpaces:
    """Test nested search spaces."""

    def test_nested_space_detection(self):
        """Test nested space is detected from _params suffix."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator_params={
                RandomForestClassifier: {
                    "n_estimators": np.arange(10, 101, 10),
                },
                SVC: {
                    "C": (0.01, 100.0, "log"),
                },
            },
        )

        assert space.has_nested_spaces
        assert "estimator" in space.dimensions  # Parent dimension created

    def test_nested_space_flattening(self):
        """Test nested space parameters are flattened with prefix."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator_params={
                RandomForestClassifier: {
                    "n_estimators": np.arange(10, 101, 10),
                },
                SVC: {
                    "C": (0.01, 100.0, "log"),
                },
            },
        )

        # Check flattened parameters exist
        assert "randomforestclassifier__n_estimators" in space.dimensions
        assert "svc__C" in space.dimensions

    def test_nested_space_auto_conditions(self):
        """Test nested space creates automatic conditions."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator_params={
                RandomForestClassifier: {
                    "n_estimators": np.arange(10, 101, 10),
                },
                SVC: {
                    "C": (0.01, 100.0, "log"),
                },
            },
        )

        # Should have conditions for each nested parameter
        assert len(space.conditions) >= 2


class TestSearchSpaceDimensionTypes:
    """Test dimension type queries."""

    def test_get_dimension_types(self):
        """Test get_dimension_types returns type mapping."""
        space = SearchSpace(
            x=np.arange(10),  # discrete
            y=(0.0, 10.0),  # continuous
            z=["a", "b"],  # categorical
        )
        types = space.get_dimension_types()
        assert types["x"] == DimensionType.DISCRETE
        assert types["y"] == DimensionType.CONTINUOUS
        assert types["z"] == DimensionType.CATEGORICAL

    def test_has_dimension_type(self):
        """Test has_dimension_type checks for type existence."""
        space = SearchSpace(x=np.arange(10), y=["a", "b"])
        assert space.has_dimension_type(DimensionType.DISCRETE)
        assert space.has_dimension_type(DimensionType.CATEGORICAL)
        assert not space.has_dimension_type(DimensionType.CONTINUOUS_LOG)


class TestSearchSpaceFiltering:
    """Test parameter filtering."""

    def test_filter_active_params(self):
        """Test filtering returns only active parameters."""
        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.1, 10.0),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        # When kernel is rbf, gamma should be active
        active = space.filter_active_params({"kernel": "rbf", "gamma": 1.0})
        assert "gamma" in active

        # When kernel is linear, gamma should be inactive
        active = space.filter_active_params({"kernel": "linear", "gamma": 1.0})
        assert "gamma" not in active


class TestSearchSpaceRepr:
    """Test string representation."""

    def test_repr_basic(self):
        """Test basic repr."""
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 10.0))
        repr_str = repr(space)
        assert "SearchSpace" in repr_str
        assert "categorical" in repr_str
        assert "continuous" in repr_str

    def test_repr_with_conditions(self):
        """Test repr includes condition count."""
        space = SearchSpace(x=[1, 2], y=[3, 4])
        space.add_condition("y", when=lambda p: p["x"] == 1)
        repr_str = repr(space)
        assert "1 conditions" in repr_str

    def test_repr_with_constraints(self):
        """Test repr includes constraint count."""
        space = SearchSpace(x=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] < 5)
        repr_str = repr(space)
        assert "1 constraints" in repr_str
