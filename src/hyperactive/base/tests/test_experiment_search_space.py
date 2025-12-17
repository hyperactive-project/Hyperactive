"""Tests for BaseExperiment search space integration.

Tests the automatic param transformation when search space is set on experiment.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.base import BaseExperiment
from hyperactive.experiment.func import FunctionExperiment
from hyperactive.search_space import SearchSpace


class TestExperimentWithoutSearchSpace:
    """Test experiment behavior when no search space is set."""

    def test_default_search_space_is_none(self):
        """Experiment should have no search space by default."""

        def simple_func(params):
            return params["x"] ** 2

        exp = FunctionExperiment(simple_func)
        assert exp._search_space is None

    def test_evaluate_passes_params_unchanged(self):
        """Without search space, params should be passed as-is."""
        received_params = {}

        def capture_func(params):
            received_params.update(params)
            return 1.0

        exp = FunctionExperiment(capture_func)
        flat_params = {"x": 10, "y": "hello"}
        exp.evaluate(flat_params)

        assert received_params == flat_params
        assert isinstance(received_params, dict)

    def test_nested_style_params_unchanged(self):
        """Prefixed params should remain flat without search space."""
        received_params = {}

        def capture_func(params):
            received_params.update(params)
            return 1.0

        exp = FunctionExperiment(capture_func)
        flat_params = {
            "estimator": "SomeClass",
            "someclass__n_estimators": 100,
            "someclass__max_depth": 5,
        }
        exp.evaluate(flat_params)

        # Params should be exactly as passed (flat)
        assert "someclass__n_estimators" in received_params
        assert received_params["someclass__n_estimators"] == 100


class TestExperimentWithSimpleSearchSpace:
    """Test experiment with SearchSpace that has no nested spaces."""

    def test_set_search_space(self):
        """Should be able to set search space via setter."""

        def simple_func(params):
            return params["x"]

        exp = FunctionExperiment(simple_func)
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 1.0))

        exp.set_search_space(space)
        assert exp._search_space is space

    def test_clear_search_space(self):
        """Should be able to clear search space by setting None."""

        def simple_func(params):
            return params["x"]

        exp = FunctionExperiment(simple_func)
        space = SearchSpace(x=[1, 2, 3])

        exp.set_search_space(space)
        exp.set_search_space(None)
        assert exp._search_space is None

    def test_simple_space_no_transformation(self):
        """Simple search space (no nesting) should not transform params."""
        received_params = {}

        def capture_func(params):
            received_params.update(params)
            return 1.0

        exp = FunctionExperiment(capture_func)
        space = SearchSpace(x=[1, 2, 3], y=(0.0, 1.0))
        exp.set_search_space(space)

        flat_params = {"x": 2, "y": 0.5}
        exp.evaluate(flat_params)

        # Without nested spaces, params should be passed as-is
        assert received_params == flat_params

    def test_space_with_conditions_no_nesting(self):
        """Space with conditions but no nesting should not transform."""
        received_params = {}

        def capture_func(params):
            received_params.update(params)
            return 1.0

        exp = FunctionExperiment(capture_func)
        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.01, 10.0, "log"),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        exp.set_search_space(space)

        flat_params = {"kernel": "rbf", "gamma": 0.1}
        exp.evaluate(flat_params)

        assert received_params == flat_params

    def test_space_with_constraints_no_nesting(self):
        """Space with constraints but no nesting should not transform."""
        received_params = {}

        def capture_func(params):
            received_params.update(params)
            return 1.0

        exp = FunctionExperiment(capture_func)
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 15)
        exp.set_search_space(space)

        flat_params = {"x": 5.0, "y": 3.0}
        exp.evaluate(flat_params)

        assert received_params == flat_params


class TestExperimentWithNestedSearchSpace:
    """Test experiment with SearchSpace that has nested spaces."""

    @pytest.fixture
    def nested_space(self):
        """Create a nested search space for testing."""

        class MockEstimatorA:
            def __init__(self, n_estimators=10, max_depth=3):
                self.n_estimators = n_estimators
                self.max_depth = max_depth

        class MockEstimatorB:
            def __init__(self, C=1.0, kernel="rbf"):
                self.C = C
                self.kernel = kernel

        space = SearchSpace(
            estimator={
                MockEstimatorA: {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [3, 5, 10],
                },
                MockEstimatorB: {
                    "C": (0.1, 10.0, "log"),
                    "kernel": ["rbf", "linear"],
                },
            },
        )
        return space, MockEstimatorA, MockEstimatorB

    def test_nested_space_transforms_params(self, nested_space):
        """Nested space should transform flat params to ParamsView."""
        space, MockEstimatorA, _ = nested_space
        received_params = None

        def capture_func(params):
            nonlocal received_params
            received_params = params
            return 1.0

        exp = FunctionExperiment(capture_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

        # Should receive ParamsView, not plain dict
        from hyperactive.search_space._params_view import ParamsView

        assert isinstance(received_params, ParamsView)

    def test_nested_params_subscript_access(self, nested_space):
        """Should be able to access nested params via subscript."""
        space, MockEstimatorA, _ = nested_space
        received_params = None

        def capture_func(params):
            nonlocal received_params
            received_params = params
            # Test nested access works
            assert params["estimator"]["n_estimators"] == 50
            assert params["estimator"]["max_depth"] == 5
            return 1.0

        exp = FunctionExperiment(capture_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

    def test_nested_params_auto_instantiate(self, nested_space):
        """Should be able to auto-instantiate via params["estimator"]()."""
        space, MockEstimatorA, _ = nested_space

        def instantiate_func(params):
            # Auto-instantiate with all nested params
            instance = params["estimator"]()
            assert isinstance(instance, MockEstimatorA)
            assert instance.n_estimators == 50
            assert instance.max_depth == 5
            return 1.0

        exp = FunctionExperiment(instantiate_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

    def test_nested_params_instantiate_with_override(self, nested_space):
        """Should be able to override params during instantiation."""
        space, MockEstimatorA, _ = nested_space

        def override_func(params):
            # Override n_estimators during instantiation
            instance = params["estimator"](n_estimators=200)
            assert isinstance(instance, MockEstimatorA)
            assert instance.n_estimators == 200  # Overridden
            assert instance.max_depth == 5  # From params
            return 1.0

        exp = FunctionExperiment(override_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

    def test_nested_params_comparison(self, nested_space):
        """Should be able to compare nested value to class."""
        space, MockEstimatorA, MockEstimatorB = nested_space

        def compare_func(params):
            assert params["estimator"] == MockEstimatorA
            assert params["estimator"] != MockEstimatorB
            return 1.0

        exp = FunctionExperiment(compare_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

    def test_nested_params_value_property(self, nested_space):
        """Should be able to get raw class via .value property."""
        space, MockEstimatorA, _ = nested_space

        def value_func(params):
            assert params["estimator"].value is MockEstimatorA
            return 1.0

        exp = FunctionExperiment(value_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

    def test_nested_params_hides_prefixed_keys(self, nested_space):
        """Prefixed keys should be hidden from iteration."""
        space, MockEstimatorA, _ = nested_space
        visible_keys = None

        def iterate_func(params):
            nonlocal visible_keys
            visible_keys = list(params.keys())
            return 1.0

        exp = FunctionExperiment(iterate_func)
        exp.set_search_space(space)

        flat_params = {
            "estimator": MockEstimatorA,
            "mockestimatora__n_estimators": 50,
            "mockestimatora__max_depth": 5,
            "mockestimatorb__C": 1.0,
            "mockestimatorb__kernel": "rbf",
        }
        exp.evaluate(flat_params)

        # Only "estimator" should be visible, not prefixed keys
        assert visible_keys == ["estimator"]


class TestExperimentSearchSpaceReuse:
    """Test reusing experiment with different search spaces."""

    def test_update_search_space(self):
        """Should be able to update search space between evaluations."""

        class EstimatorX:
            def __init__(self, alpha=1.0):
                self.alpha = alpha

        class EstimatorY:
            def __init__(self, beta=1.0):
                self.beta = beta

        space1 = SearchSpace(
            estimator={
                EstimatorX: {"alpha": [0.1, 1.0, 10.0]},
            },
        )
        space2 = SearchSpace(
            estimator={
                EstimatorY: {"beta": [0.1, 1.0, 10.0]},
            },
        )

        results = []

        def capture_func(params):
            results.append(params["estimator"].value.__name__)
            return 1.0

        exp = FunctionExperiment(capture_func)

        # First evaluation with space1
        exp.set_search_space(space1)
        exp.evaluate({"estimator": EstimatorX, "estimatorx__alpha": 1.0})

        # Second evaluation with space2
        exp.set_search_space(space2)
        exp.evaluate({"estimator": EstimatorY, "estimatory__beta": 1.0})

        assert results == ["EstimatorX", "EstimatorY"]

    def test_switch_between_nested_and_simple(self):
        """Should handle switching between nested and simple spaces."""

        class EstimatorZ:
            def __init__(self, gamma=1.0):
                self.gamma = gamma

        nested_space = SearchSpace(
            estimator={
                EstimatorZ: {"gamma": [0.1, 1.0]},
            },
        )
        simple_space = SearchSpace(x=[1, 2, 3], y=(0.0, 1.0))

        param_types = []

        def capture_type(params):
            param_types.append(type(params).__name__)
            return 1.0

        exp = FunctionExperiment(capture_type)

        # Nested space -> ParamsView
        exp.set_search_space(nested_space)
        exp.evaluate({"estimator": EstimatorZ, "estimatorz__gamma": 1.0})

        # Simple space -> dict (no transformation)
        exp.set_search_space(simple_space)
        exp.evaluate({"x": 1, "y": 0.5})

        # No space -> dict
        exp.set_search_space(None)
        exp.evaluate({"a": 1, "b": 2})

        assert param_types == ["ParamsView", "dict", "dict"]


class TestOptimizerExperimentIntegration:
    """Test that optimizer correctly sets search space on experiment."""

    def test_optimizer_sets_search_space(self):
        """Optimizer should set search space on experiment before solve."""

        class MockEstimator:
            def __init__(self, n=10):
                self.n = n

        space = SearchSpace(
            estimator={
                MockEstimator: {"n": [10, 20, 30]},
            },
        )

        instantiated = []

        def objective(params):
            instance = params["estimator"]()
            instantiated.append(instance.n)
            return float(instance.n)

        from hyperactive.opt import RandomSearch

        opt = RandomSearch(
            search_space=space,
            n_iter=3,
            experiment=objective,
        )
        opt.solve()

        # Should have instantiated at least once (GFO caches duplicates)
        assert len(instantiated) >= 1
        # All values should be from the search space
        assert all(n in [10, 20, 30] for n in instantiated)

    def test_optimizer_with_dict_space_no_transformation(self):
        """Optimizer with dict space should not transform params."""
        received_types = []

        def objective(params):
            received_types.append(type(params).__name__)
            return params["x"]

        from hyperactive.opt import RandomSearch

        # Using dict instead of SearchSpace (backward compatibility)
        opt = RandomSearch(
            search_space={"x": np.array([1, 2, 3])},
            n_iter=3,
            experiment=objective,
        )
        opt.solve()

        # Should receive plain dict, not ParamsView
        assert all(t == "dict" for t in received_types)


class TestConfigureExperimentBehavior:
    """Test _configure_experiment behavior across different optimizer types.

    _configure_experiment is called after get_search_config(), so for adapters
    that convert SearchSpace to dict (like GFO), it effectively does nothing.
    This tests that the method correctly handles different input types.
    """

    def test_configure_experiment_with_dict_space_does_nothing(self):
        """_configure_experiment should return early for dict search_space."""
        from unittest.mock import Mock

        from hyperactive.base import BaseOptimizer

        # Create a minimal concrete optimizer
        class TestOptimizer(BaseOptimizer):
            def __init__(self, search_space=None, experiment=None):
                self.search_space = search_space
                self.experiment = experiment
                super().__init__()

            def _solve(self, experiment, **kwargs):
                return {}

        experiment = Mock()
        experiment.set_search_space = Mock()

        opt = TestOptimizer(
            search_space={"x": [1, 2, 3]},
            experiment=lambda p: 1.0,
        )

        # Call _configure_experiment with dict search_space
        search_config = {"search_space": {"x": [1, 2, 3]}}
        opt._configure_experiment(experiment, search_config)

        # set_search_space should NOT be called for dict
        experiment.set_search_space.assert_not_called()

    def test_configure_experiment_with_none_space_does_nothing(self):
        """_configure_experiment should return early for None search_space."""
        from unittest.mock import Mock

        from hyperactive.base import BaseOptimizer

        class TestOptimizer(BaseOptimizer):
            def __init__(self, search_space=None, experiment=None):
                self.search_space = search_space
                self.experiment = experiment
                super().__init__()

            def _solve(self, experiment, **kwargs):
                return {}

        experiment = Mock()
        experiment.set_search_space = Mock()

        opt = TestOptimizer(experiment=lambda p: 1.0)

        search_config = {"search_space": None}
        opt._configure_experiment(experiment, search_config)

        experiment.set_search_space.assert_not_called()

    def test_configure_experiment_with_searchspace_calls_set_and_validate(self):
        """_configure_experiment should set and validate for SearchSpace."""
        from unittest.mock import Mock, patch

        from hyperactive.base import BaseOptimizer

        class TestOptimizer(BaseOptimizer):
            _tags = {
                "capability:search_space:conditional": False,
                "capability:search_space:constraints": False,
                "capability:search_space:nested": False,
            }

            def __init__(self, search_space=None, experiment=None):
                self.search_space = search_space
                self.experiment = experiment
                super().__init__()

            def _solve(self, experiment, **kwargs):
                return {}

        experiment = Mock()
        experiment.set_search_space = Mock()

        space = SearchSpace(x=[1, 2, 3])
        opt = TestOptimizer(
            search_space=space,
            experiment=lambda p: 1.0,
        )

        # Pass SearchSpace directly (not converted to dict)
        search_config = {"search_space": space}
        opt._configure_experiment(experiment, search_config)

        # set_search_space SHOULD be called for SearchSpace
        experiment.set_search_space.assert_called_once_with(space)

    def test_gfo_adapter_converts_searchspace_before_configure_experiment(self):
        """GFO adapter converts SearchSpace to dict before _configure_experiment.

        This test verifies that for GFO, _configure_experiment receives a dict,
        not a SearchSpace, because get_search_config() converts it first.
        """
        from hyperactive.opt import HillClimbing

        space = SearchSpace(x=[1, 2, 3])
        opt = HillClimbing(
            search_space=space,
            n_iter=1,
            experiment=lambda p: 1.0,
        )

        # Get the search config (this is what _configure_experiment receives)
        search_config = opt.get_search_config()

        # Verify search_space is now a dict, not SearchSpace
        assert isinstance(search_config["search_space"], dict)
        assert not isinstance(search_config["search_space"], SearchSpace)


class TestAdapterValidation:
    """Test that each adapter validates SearchSpace features correctly."""

    def test_gfo_adapter_validates_before_conversion(self):
        """GFO adapter should validate SearchSpace before converting to dict."""
        import warnings

        from hyperactive.opt import HillClimbing

        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.1, 10.0),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        opt = HillClimbing(
            search_space=space,
            n_iter=1,
            experiment=lambda p: 1.0,
        )

        # Validation happens in get_search_config
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt.get_search_config()

            condition_warnings = [
                x for x in w if "conditional" in str(x.message).lower()
            ]
            assert len(condition_warnings) >= 1

    def test_sklearn_adapter_validates_before_conversion(self):
        """sklearn adapter should validate SearchSpace before converting."""
        import warnings

        from hyperactive.opt import RandomSearchSk

        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(0.1, 10.0),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        opt = RandomSearchSk(
            param_distributions=space,
            n_iter=1,
            experiment=lambda p: 1.0,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Validation happens during solve (in _solve -> _convert_param_distributions)
            opt.solve()

            condition_warnings = [
                x for x in w if "conditional" in str(x.message).lower()
            ]
            assert len(condition_warnings) >= 1

    def test_no_validation_for_dict_space(self):
        """No validation should occur for plain dict search spaces."""
        import warnings

        from hyperactive.opt import HillClimbing

        # Plain dict, not SearchSpace
        search_space = {"x": np.array([1, 2, 3])}

        opt = HillClimbing(
            search_space=search_space,
            n_iter=1,
            experiment=lambda p: 1.0,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt.get_search_config()

            # No warnings about features since dict has no feature metadata
            search_space_warnings = [
                x
                for x in w
                if "conditional" in str(x.message).lower()
                or "constraint" in str(x.message).lower()
                or "nested" in str(x.message).lower()
            ]
            assert len(search_space_warnings) == 0

    def test_validation_warning_includes_optimizer_name(self):
        """Validation warnings should include the optimizer name."""
        import warnings

        from hyperactive.opt import HillClimbing

        space = SearchSpace(x=[1, 2, 3])
        space.add_condition("x", when=lambda p: True)

        opt = HillClimbing(
            search_space=space,
            n_iter=1,
            experiment=lambda p: 1.0,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt.get_search_config()

            condition_warnings = [
                x for x in w if "conditional" in str(x.message).lower()
            ]
            assert len(condition_warnings) >= 1
            # Should mention "Hill Climbing" in the warning
            assert "Hill Climbing" in str(condition_warnings[0].message)


class TestBackwardCompatibilityWithDictSpace:
    """Test backward compatibility when using dict instead of SearchSpace."""

    def test_gfo_with_dict_backward_compatible(self):
        """GFO should work with plain dict (backward compatibility)."""
        from hyperactive.opt import HillClimbing

        search_space = {"x": np.array([1, 2, 3])}

        def objective(params):
            return float(params["x"])

        opt = HillClimbing(
            search_space=search_space,
            n_iter=5,
            experiment=objective,
        )

        result = opt.solve()
        assert "x" in result
        assert result["x"] in [1, 2, 3]

    def test_sklearn_with_dict_backward_compatible(self):
        """sklearn should work with plain dict (backward compatibility)."""
        from hyperactive.opt import RandomSearchSk

        param_distributions = {"x": [1, 2, 3], "y": [4, 5, 6]}

        def objective(params):
            return float(params["x"] + params["y"])

        opt = RandomSearchSk(
            param_distributions=param_distributions,
            n_iter=5,
            experiment=objective,
        )

        result = opt.solve()
        assert "x" in result
        assert "y" in result
