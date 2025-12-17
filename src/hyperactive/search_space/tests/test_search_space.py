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
        """Test nested space is detected from class keys in dict."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
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
        assert "estimator" in space.nested_spaces  # Nested space stored

    def test_nested_space_flattening(self):
        """Test nested space parameters are flattened with prefix."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
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
            estimator={
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

    def test_lambda_collision_raises_error(self):
        """Test that multiple lambdas as nested space keys raises ValueError.

        Lambda functions all have __name__ == '<lambda>', so they would produce
        the same prefix and cause parameter name collisions.
        """
        with pytest.raises(ValueError, match="same prefix"):
            SearchSpace(
                transform={
                    lambda x: x**2: {"power": [2, 3]},
                    lambda x: x + 1: {"offset": [1, 2]},
                }
            )

    def test_duplicate_prefix_from_same_name_raises(self):
        """Test that keys producing the same prefix are detected."""

        def my_func_a():
            pass

        def my_func_b():
            pass

        # Rename to have same __name__
        my_func_b.__name__ = "my_func_a"

        with pytest.raises(ValueError, match="same prefix"):
            SearchSpace(
                transform={
                    my_func_a: {"x": [1, 2]},
                    my_func_b: {"y": [3, 4]},
                }
            )


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


class TestMultipleLevelNesting:
    """Test multiple levels of nesting in search spaces."""

    def test_two_level_nesting_structure(self):
        """Test two-level nesting creates correct structure.

        Structure: estimator -> {RFC, SVC} -> {params for each}
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
                RandomForestClassifier: {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [3, 5, 10],
                },
                SVC: {
                    "C": (0.1, 10.0, "log"),
                    "kernel": ["rbf", "linear"],
                },
            },
        )

        # Verify parent dimension
        assert "estimator" in space.dimensions
        assert space.dimensions["estimator"].dim_type == DimensionType.CATEGORICAL
        assert set(space.dimensions["estimator"].values) == {
            RandomForestClassifier,
            SVC,
        }

        # Verify flattened RFC parameters
        assert "randomforestclassifier__n_estimators" in space.dimensions
        assert "randomforestclassifier__max_depth" in space.dimensions
        rfc_n_est = space.dimensions["randomforestclassifier__n_estimators"]
        assert rfc_n_est.values == [10, 50, 100]
        assert rfc_n_est.dim_type == DimensionType.CATEGORICAL

        # Verify flattened SVC parameters
        assert "svc__C" in space.dimensions
        assert "svc__kernel" in space.dimensions
        svc_c = space.dimensions["svc__C"]
        assert svc_c.dim_type == DimensionType.CONTINUOUS_LOG

    def test_two_level_nesting_conditions(self):
        """Test two-level nesting creates correct conditions.

        Each child parameter should only be active when parent matches.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
                RandomForestClassifier: {"n_estimators": [10, 50]},
                SVC: {"C": [0.1, 1.0]},
            },
        )

        # Should have 2 conditions (one for each child param)
        assert len(space.conditions) == 2

        # Find conditions by target
        rfc_condition = next(
            c
            for c in space.conditions
            if c.target_param == "randomforestclassifier__n_estimators"
        )
        svc_condition = next(
            c for c in space.conditions if c.target_param == "svc__C"
        )

        # RFC param should be active only when estimator=RFC
        assert rfc_condition.is_active({"estimator": RandomForestClassifier})
        assert not rfc_condition.is_active({"estimator": SVC})

        # SVC param should be active only when estimator=SVC
        assert svc_condition.is_active({"estimator": SVC})
        assert not svc_condition.is_active({"estimator": RandomForestClassifier})

    def test_multiple_nested_parents(self):
        """Test multiple independent nested spaces at same level.

        Structure:
        - estimator -> {RFC, SVC}
        - scaler -> {StandardScaler, MinMaxScaler}
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
                RandomForestClassifier: {"n_estimators": [10, 50]},
                SVC: {"C": [0.1, 1.0]},
            },
            scaler={
                StandardScaler: {"with_mean": [True, False]},
                MinMaxScaler: {"feature_range": [(0, 1), (-1, 1)]},
            },
        )

        # Should have 2 nested spaces
        assert len(space.nested_spaces) == 2
        assert "estimator" in space.nested_spaces
        assert "scaler" in space.nested_spaces

        # Both parent dimensions should exist
        assert "estimator" in space.dimensions
        assert "scaler" in space.dimensions

        # All child dimensions should exist with correct prefixes
        assert "randomforestclassifier__n_estimators" in space.dimensions
        assert "svc__C" in space.dimensions
        assert "standardscaler__with_mean" in space.dimensions
        assert "minmaxscaler__feature_range" in space.dimensions

        # Should have 4 conditions (one per child param)
        assert len(space.conditions) == 4

    def test_nested_space_filter_active_params(self):
        """Test filter_active_params works with nested spaces."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            estimator={
                RandomForestClassifier: {"n_estimators": [10, 50]},
                SVC: {"C": [0.1, 1.0]},
            },
        )

        # When estimator is RFC, only RFC params should be active
        params_rfc = {
            "estimator": RandomForestClassifier,
            "randomforestclassifier__n_estimators": 50,
            "svc__C": 1.0,  # This should be filtered out
        }
        active = space.filter_active_params(params_rfc)

        assert "estimator" in active
        assert "randomforestclassifier__n_estimators" in active
        assert "svc__C" not in active  # Filtered because estimator != SVC

        # When estimator is SVC, only SVC params should be active
        params_svc = {
            "estimator": SVC,
            "randomforestclassifier__n_estimators": 50,  # Should be filtered
            "svc__C": 1.0,
        }
        active = space.filter_active_params(params_svc)

        assert "estimator" in active
        assert "svc__C" in active
        assert "randomforestclassifier__n_estimators" not in active

    def test_callable_keys_not_just_classes(self):
        """Test nested spaces work with callable keys (not just classes)."""

        def model_a(x):
            return x * 2

        def model_b(x):
            return x + 1

        space = SearchSpace(
            model={
                model_a: {"factor": [1, 2, 3]},
                model_b: {"offset": [0, 1, 2]},
            },
        )

        assert space.has_nested_spaces
        assert "model" in space.dimensions
        assert set(space.dimensions["model"].values) == {model_a, model_b}

        # Verify flattened params use function names
        assert "model_a__factor" in space.dimensions
        assert "model_b__offset" in space.dimensions

    def test_nested_space_count(self):
        """Test total dimension count with nested spaces."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        space = SearchSpace(
            lr=(1e-5, 1e-1, "log"),  # 1 regular param
            estimator={  # 1 parent + 3 child params
                RandomForestClassifier: {
                    "n_estimators": [10, 50],
                    "max_depth": [3, 5],
                },
                SVC: {"C": [0.1, 1.0]},
            },
        )

        # Total: lr + estimator + rfc__n_est + rfc__max_depth + svc__C = 5
        assert len(space) == 5
        assert len(space.param_names) == 5


class TestConditionsComprehensive:
    """Comprehensive tests for conditional search spaces."""

    def test_condition_is_active_true(self):
        """Test condition returns True when predicate is satisfied."""
        space = SearchSpace(
            kernel=["rbf", "linear", "poly"],
            gamma=(0.01, 10.0, "log"),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        condition = space.conditions[0]
        assert condition.is_active({"kernel": "rbf"}) is True
        assert condition.is_active({"kernel": "linear"}) is False
        assert condition.is_active({"kernel": "poly"}) is False

    def test_condition_is_active_with_missing_dependency(self):
        """Test condition returns False when dependency is missing."""
        space = SearchSpace(kernel=["rbf", "linear"], gamma=(0.01, 10.0))
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        condition = space.conditions[0]
        # Missing "kernel" key should return False (safe default)
        assert condition.is_active({}) is False
        assert condition.is_active({"other": "value"}) is False

    def test_condition_can_evaluate(self):
        """Test can_evaluate checks for dependency presence."""
        space = SearchSpace(kernel=["rbf", "linear"], gamma=(0.01, 10.0))
        space.add_condition(
            "gamma",
            when=lambda p: p["kernel"] == "rbf",
            depends_on="kernel",
        )

        condition = space.conditions[0]
        assert condition.can_evaluate({"kernel": "rbf"}) is True
        assert condition.can_evaluate({"kernel": "linear", "gamma": 1.0}) is True
        assert condition.can_evaluate({}) is False
        assert condition.can_evaluate({"gamma": 1.0}) is False

    def test_multiple_conditions_on_same_param(self):
        """Test multiple conditions can control the same parameter.

        All conditions must be True for param to be active.
        """
        space = SearchSpace(
            mode=["simple", "advanced"],
            level=[1, 2, 3],
            extra_param=[10, 20, 30],
        )
        # extra_param active only in advanced mode AND level >= 2
        space.add_condition("extra_param", when=lambda p: p["mode"] == "advanced")
        space.add_condition("extra_param", when=lambda p: p["level"] >= 2)

        # Both conditions must be met
        active = space.filter_active_params(
            {"mode": "advanced", "level": 2, "extra_param": 10}
        )
        assert "extra_param" in active

        # Only mode condition met
        active = space.filter_active_params(
            {"mode": "advanced", "level": 1, "extra_param": 10}
        )
        assert "extra_param" not in active

        # Only level condition met
        active = space.filter_active_params(
            {"mode": "simple", "level": 3, "extra_param": 10}
        )
        assert "extra_param" not in active

        # Neither condition met
        active = space.filter_active_params(
            {"mode": "simple", "level": 1, "extra_param": 10}
        )
        assert "extra_param" not in active

    def test_chained_conditions(self):
        """Test conditions that depend on other conditional params.

        Structure: A -> B -> C (if A, then B available; if B, then C available)

        Note: The current implementation evaluates conditions independently
        against the raw params dict. It does NOT support transitive conditions
        where a param's activity depends on whether another conditional param
        is active. Each condition is evaluated against the param values directly.
        """
        space = SearchSpace(
            enable_feature=[True, False],
            feature_type=["basic", "advanced"],
            advanced_setting=[1, 2, 3],
        )
        # feature_type only available if enable_feature=True
        space.add_condition(
            "feature_type",
            when=lambda p: p["enable_feature"] is True,
            depends_on="enable_feature",
        )
        # advanced_setting only available if feature_type="advanced"
        space.add_condition(
            "advanced_setting",
            when=lambda p: p.get("feature_type") == "advanced",
            depends_on="feature_type",
        )

        # All enabled: feature=True, type=advanced -> advanced_setting active
        active = space.filter_active_params(
            {
                "enable_feature": True,
                "feature_type": "advanced",
                "advanced_setting": 2,
            }
        )
        assert "enable_feature" in active
        assert "feature_type" in active
        assert "advanced_setting" in active

        # Feature enabled but basic type -> no advanced_setting
        active = space.filter_active_params(
            {
                "enable_feature": True,
                "feature_type": "basic",
                "advanced_setting": 2,
            }
        )
        assert "enable_feature" in active
        assert "feature_type" in active
        assert "advanced_setting" not in active

        # Feature disabled -> feature_type inactive
        # Note: advanced_setting is STILL active because its condition
        # checks feature_type VALUE (which is "advanced"), not whether
        # feature_type is ACTIVE. This is current implementation behavior.
        active = space.filter_active_params(
            {
                "enable_feature": False,
                "feature_type": "advanced",
                "advanced_setting": 2,
            }
        )
        assert "enable_feature" in active
        assert "feature_type" not in active
        # advanced_setting remains active because its condition evaluates True
        # (feature_type=="advanced" in the raw params)
        assert "advanced_setting" in active

    def test_chained_conditions_proper_filtering(self):
        """Test proper chain filtering by combining conditions.

        To achieve true transitive conditions, combine the checks in
        a single predicate that evaluates all upstream dependencies.
        """
        space = SearchSpace(
            enable_feature=[True, False],
            feature_type=["basic", "advanced"],
            advanced_setting=[1, 2, 3],
        )
        space.add_condition(
            "feature_type",
            when=lambda p: p["enable_feature"] is True,
        )
        # For true chaining, check BOTH conditions in the predicate
        space.add_condition(
            "advanced_setting",
            when=lambda p: p["enable_feature"] is True
            and p.get("feature_type") == "advanced",
        )

        # All enabled -> advanced_setting active
        active = space.filter_active_params(
            {"enable_feature": True, "feature_type": "advanced", "advanced_setting": 2}
        )
        assert "advanced_setting" in active

        # Feature disabled -> advanced_setting now properly inactive
        active = space.filter_active_params(
            {"enable_feature": False, "feature_type": "advanced", "advanced_setting": 2}
        )
        assert "advanced_setting" not in active

    def test_condition_with_complex_predicate(self):
        """Test condition with complex multi-param predicate."""
        space = SearchSpace(
            x=(0.0, 10.0),
            y=(0.0, 10.0),
            special_param=[1, 2, 3],
        )
        # special_param only active when x + y > 5 AND x > y
        space.add_condition(
            "special_param",
            when=lambda p: (p["x"] + p["y"]) > 5 and p["x"] > p["y"],
            depends_on=["x", "y"],
        )

        # x=4, y=2: sum=6>5, x>y -> active
        active = space.filter_active_params({"x": 4, "y": 2, "special_param": 1})
        assert "special_param" in active

        # x=3, y=1: sum=4<5 -> inactive
        active = space.filter_active_params({"x": 3, "y": 1, "special_param": 1})
        assert "special_param" not in active

        # x=2, y=4: x<y -> inactive
        active = space.filter_active_params({"x": 2, "y": 4, "special_param": 1})
        assert "special_param" not in active

    def test_condition_name_tracking(self):
        """Test conditions can be named for debugging."""
        space = SearchSpace(kernel=["rbf", "linear"], gamma=(0.01, 10.0))
        space.add_condition(
            "gamma",
            when=lambda p: p["kernel"] == "rbf",
            name="gamma_rbf_only",
        )

        condition = space.conditions[0]
        assert condition.name == "gamma_rbf_only"
        assert "gamma_rbf_only" in repr(condition)

    def test_conditions_preserved_in_union(self):
        """Test conditions are preserved when unioning spaces."""
        s1 = SearchSpace(kernel=["rbf", "linear"], gamma=(0.01, 10.0))
        s1.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        s2 = SearchSpace(C=(0.1, 100.0), shrinking=[True, False])
        s2.add_condition("shrinking", when=lambda p: p["C"] > 1.0)

        combined = s1 | s2

        # Both conditions should be preserved
        assert len(combined.conditions) == 2
        targets = {c.target_param for c in combined.conditions}
        assert targets == {"gamma", "shrinking"}


class TestConstraintsComprehensive:
    """Comprehensive tests for search space constraints."""

    def test_constraint_is_satisfied(self):
        """Test constraint satisfaction checking."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 15)

        # Satisfied: 3 + 4 = 7 < 15
        assert space.check_constraints({"x": 3, "y": 4}) is True

        # Satisfied: 7 + 7 = 14 < 15
        assert space.check_constraints({"x": 7, "y": 7}) is True

        # Not satisfied: 8 + 8 = 16 >= 15
        assert space.check_constraints({"x": 8, "y": 8}) is False

    def test_multiple_constraints(self):
        """Test multiple constraints all must be satisfied."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0), z=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 10, name="sum_xy")
        space.add_constraint(lambda p: p["y"] + p["z"] < 10, name="sum_yz")
        space.add_constraint(lambda p: p["x"] < p["z"], name="x_lt_z")

        # All constraints satisfied: x=2, y=3, z=5
        # sum_xy: 5 < 10 ✓, sum_yz: 8 < 10 ✓, x_lt_z: 2 < 5 ✓
        assert space.check_constraints({"x": 2, "y": 3, "z": 5}) is True

        # Fails sum_xy: x=6, y=6, z=7
        assert space.check_constraints({"x": 6, "y": 6, "z": 7}) is False

        # Fails x_lt_z: x=5, y=2, z=3
        assert space.check_constraints({"x": 5, "y": 2, "z": 3}) is False

    def test_constraint_with_categorical(self):
        """Test constraints can use categorical params."""
        space = SearchSpace(
            mode=["low", "medium", "high"],
            intensity=(0.0, 1.0),
        )
        # intensity must be low when mode is "low"
        space.add_constraint(
            lambda p: p["intensity"] <= 0.3 if p["mode"] == "low" else True
        )

        # mode=low, intensity=0.2 -> valid
        assert space.check_constraints({"mode": "low", "intensity": 0.2}) is True

        # mode=low, intensity=0.5 -> invalid
        assert space.check_constraints({"mode": "low", "intensity": 0.5}) is False

        # mode=high, intensity=0.9 -> valid (constraint doesn't apply)
        assert space.check_constraints({"mode": "high", "intensity": 0.9}) is True

    def test_constraint_with_missing_param(self):
        """Test constraint returns False when param is missing."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 10)

        # Missing x -> should return False (cannot evaluate)
        assert space.check_constraints({"y": 5}) is False

        # Missing y -> should return False
        assert space.check_constraints({"x": 5}) is False

        # Both missing -> should return False
        assert space.check_constraints({}) is False

    def test_constraint_name_auto_generated(self):
        """Test constraints get auto-generated names if not provided."""
        space = SearchSpace(x=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] < 5)
        space.add_constraint(lambda p: p["x"] > 1)

        assert space.constraints[0].name == "constraint_0"
        assert space.constraints[1].name == "constraint_1"

    def test_constraint_with_explicit_name(self):
        """Test constraints can have explicit names."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(
            lambda p: p["x"] * p["y"] < 50,
            name="product_limit",
        )

        assert space.constraints[0].name == "product_limit"

    def test_constraint_callable_directly(self):
        """Test Constraint object can be called directly like a function."""
        from hyperactive.search_space._constraint import Constraint

        constraint = Constraint(
            predicate=lambda p: p["x"] > 0,
            name="positive_x",
        )

        # Can call constraint directly
        assert constraint({"x": 5}) is True
        assert constraint({"x": -1}) is False
        assert constraint({"x": 0}) is False

    def test_constraints_preserved_in_union(self):
        """Test constraints are preserved when unioning spaces."""
        s1 = SearchSpace(x=(0.0, 10.0))
        s1.add_constraint(lambda p: p["x"] < 5, name="x_limit")

        s2 = SearchSpace(y=(0.0, 10.0))
        s2.add_constraint(lambda p: p["y"] > 2, name="y_min")

        combined = s1 | s2

        # Both constraints should be preserved
        assert len(combined.constraints) == 2
        names = {c.name for c in combined.constraints}
        assert names == {"x_limit", "y_min"}

    def test_constraint_with_params_metadata(self):
        """Test constraints can track which params they involve."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0), z=(0.0, 10.0))
        space.add_constraint(
            lambda p: p["x"] + p["y"] < 10,
            name="sum_constraint",
            params=["x", "y"],
        )

        constraint = space.constraints[0]
        assert constraint.params == ["x", "y"]

    def test_empty_constraints_check(self):
        """Test check_constraints returns True when no constraints."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))

        # No constraints added -> should return True for any params
        assert space.check_constraints({"x": 100, "y": 100}) is True
        assert space.check_constraints({}) is True


class TestConditionsAndConstraintsTogether:
    """Test interactions between conditions and constraints."""

    def test_conditions_and_constraints_independent(self):
        """Test conditions and constraints are evaluated independently."""
        space = SearchSpace(
            mode=["simple", "advanced"],
            x=(0.0, 10.0),
            advanced_param=[1, 2, 3],
        )
        # Condition: advanced_param only in advanced mode
        space.add_condition(
            "advanced_param", when=lambda p: p["mode"] == "advanced"
        )
        # Constraint: x must be less than 5
        space.add_constraint(lambda p: p["x"] < 5)

        # Valid: advanced mode, x=3, advanced_param=2
        params = {"mode": "advanced", "x": 3, "advanced_param": 2}
        assert space.check_constraints(params) is True
        active = space.filter_active_params(params)
        assert "advanced_param" in active

        # Invalid by constraint: x=6
        params = {"mode": "advanced", "x": 6, "advanced_param": 2}
        assert space.check_constraints(params) is False

        # Valid but conditional param inactive
        params = {"mode": "simple", "x": 3, "advanced_param": 2}
        assert space.check_constraints(params) is True
        active = space.filter_active_params(params)
        assert "advanced_param" not in active

    def test_constraint_on_conditional_param(self):
        """Test constraint can reference a conditional parameter."""
        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.01, 10.0),
            C=(0.1, 100.0),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        # Constraint: gamma * C must be < 100 (only meaningful when gamma exists)
        space.add_constraint(
            lambda p: p.get("gamma", 1.0) * p["C"] < 100,
            name="gamma_C_limit",
        )

        # kernel=rbf, gamma=0.5, C=10 -> gamma*C=5 < 100 ✓
        assert space.check_constraints({"kernel": "rbf", "gamma": 0.5, "C": 10})

        # kernel=rbf, gamma=5, C=50 -> gamma*C=250 >= 100 ✗
        assert not space.check_constraints({"kernel": "rbf", "gamma": 5, "C": 50})

        # kernel=linear (gamma inactive), C=50 -> uses default gamma=1.0, 50 < 100 ✓
        assert space.check_constraints({"kernel": "linear", "C": 50})
