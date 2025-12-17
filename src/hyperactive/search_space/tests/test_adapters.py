"""Tests for search space adapters."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.search_space import SearchSpace
from hyperactive.search_space.adapters import (
    GFOSearchSpaceAdapter,
    SklearnSearchSpaceAdapter,
    get_adapter,
)


class TestGFOAdapter:
    """Test GFO backend adaptation."""

    def test_categorical_to_object_array(self, simple_space):
        """Test categorical dimensions become object arrays."""
        adapted = simple_space.to_backend("gfo")
        assert adapted["kernel"].dtype == object
        assert list(adapted["kernel"]) == ["rbf", "linear", "poly"]

    def test_continuous_discretized(self):
        """Test continuous dimensions are discretized."""
        space = SearchSpace(x=(0.0, 10.0))
        adapted = space.to_backend("gfo", resolution=50)
        assert len(adapted["x"]) == 50
        assert adapted["x"][0] == 0.0
        assert adapted["x"][-1] == 10.0

    def test_log_scale_uses_logspace(self):
        """Test log-scale uses numpy logspace."""
        space = SearchSpace(lr=(1e-5, 1e-1, "log"))
        adapted = space.to_backend("gfo", resolution=10)
        # Check logarithmic spacing (equal ratios)
        ratios = adapted["lr"][1:] / adapted["lr"][:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-5)

    def test_constant_becomes_single_element_array(self):
        """Test constants become single-element arrays."""
        space = SearchSpace(seed=42)
        adapted = space.to_backend("gfo")
        assert len(adapted["seed"]) == 1
        assert adapted["seed"][0] == 42

    def test_classes_preserved_in_categorical(self):
        """Test class objects are preserved in categorical."""
        from sklearn.ensemble import RandomForestClassifier

        space = SearchSpace(est=[RandomForestClassifier])
        adapted = space.to_backend("gfo")
        assert adapted["est"][0] is RandomForestClassifier

    def test_discrete_passthrough(self):
        """Test discrete arrays pass through unchanged."""
        arr = np.array([1, 2, 3, 5, 8])
        space = SearchSpace(x=arr)
        adapted = space.to_backend("gfo")
        np.testing.assert_array_equal(adapted["x"], arr)

    def test_scipy_distribution_sampled(self):
        """Test scipy distributions are sampled."""
        import scipy.stats as st

        space = SearchSpace(x=st.uniform(0, 1))
        adapted = space.to_backend("gfo", resolution=50)
        assert len(adapted["x"]) > 0
        assert all(0 <= v <= 1 for v in adapted["x"])


class TestSklearnAdapter:
    """Test sklearn backend adaptation."""

    def test_categorical_to_list(self):
        """Test categorical becomes list."""
        space = SearchSpace(kernel=["rbf", "linear"])
        adapted = space.to_backend("sklearn_random")
        assert adapted["kernel"] == ["rbf", "linear"]

    def test_scipy_distribution_passthrough(self):
        """Test scipy distributions pass through."""
        import scipy.stats as st

        dist = st.uniform(0, 1)
        space = SearchSpace(x=dist)
        adapted = space.to_backend("sklearn_random")
        assert adapted["x"] is dist

    def test_continuous_to_scipy_uniform(self):
        """Test continuous becomes scipy uniform."""
        space = SearchSpace(x=(0.0, 10.0))
        adapted = space.to_backend("sklearn_random")
        assert hasattr(adapted["x"], "rvs")  # Is a scipy distribution

    def test_log_scale_to_loguniform(self):
        """Test log-scale becomes loguniform."""
        space = SearchSpace(lr=(1e-5, 1e-1, "log"))
        adapted = space.to_backend("sklearn_random")
        assert hasattr(adapted["lr"], "rvs")
        # Sample and check range
        samples = adapted["lr"].rvs(size=100)
        assert all(1e-5 <= s <= 1e-1 for s in samples)

    def test_grid_mode(self):
        """Test grid mode returns lists."""
        adapter = SklearnSearchSpaceAdapter(
            SearchSpace(x=(0.0, 10.0), y=["a", "b"])
        )
        grid = adapter.adapt(mode="grid", grid_resolution=5)
        assert isinstance(grid["x"], list)
        assert isinstance(grid["y"], list)
        assert len(grid["x"]) == 5


class TestOptunaAdapter:
    """Test Optuna backend adaptation."""

    @pytest.fixture
    def skip_if_no_optuna(self):
        """Skip test if optuna is not available."""
        pytest.importorskip("optuna")

    def test_continuous_to_tuple(self, skip_if_no_optuna):
        """Test continuous becomes tuple."""
        space = SearchSpace(x=(0.0, 10.0))
        adapted = space.to_backend("optuna")
        assert adapted["x"] == (0.0, 10.0)

    def test_categorical_to_list(self, skip_if_no_optuna):
        """Test categorical becomes list."""
        space = SearchSpace(kernel=["rbf", "linear"])
        adapted = space.to_backend("optuna")
        assert adapted["kernel"] == ["rbf", "linear"]

    def test_log_scale_to_distribution(self, skip_if_no_optuna):
        """Test log-scale becomes Optuna distribution."""
        import optuna

        space = SearchSpace(lr=(1e-5, 1e-1, "log"))
        adapted = space.to_backend("optuna")
        assert isinstance(adapted["lr"], optuna.distributions.FloatDistribution)
        assert adapted["lr"].log is True


class TestAdapterFactory:
    """Test get_adapter factory function."""

    def test_get_gfo_adapter(self):
        """Test getting GFO adapter."""
        space = SearchSpace(x=[1, 2, 3])
        adapter = get_adapter("gfo", space)
        assert isinstance(adapter, GFOSearchSpaceAdapter)

    def test_get_sklearn_adapter(self):
        """Test getting sklearn adapter."""
        space = SearchSpace(x=[1, 2, 3])
        adapter = get_adapter("sklearn", space)
        assert isinstance(adapter, SklearnSearchSpaceAdapter)

    def test_unknown_backend_raises(self):
        """Test unknown backend raises error."""
        space = SearchSpace(x=[1, 2, 3])
        with pytest.raises(ValueError, match="Unknown backend"):
            get_adapter("unknown", space)


class TestConstraintHandling:
    """Test constraint extraction from SearchSpace."""

    def test_gfo_constraints_extracted(self):
        """Test GFO adapter extracts constraints."""
        space = SearchSpace(x=(0.0, 10.0), y=(0.0, 10.0))
        space.add_constraint(lambda p: p["x"] + p["y"] < 10)

        adapter = GFOSearchSpaceAdapter(space)
        constraints = adapter.get_constraints()

        assert constraints is not None
        assert len(constraints) == 1  # Exactly 1 constraint, no conditions

    def test_gfo_conditions_not_converted_to_constraints(self):
        """Test GFO adapter does NOT convert conditions to constraints.

        This is critical: conditions should not become constraints because
        GFO samples all parameters. If conditions were constraints, they would
        incorrectly reject valid parameter combinations where conditional
        parameters happen to be sampled but aren't "active".
        """
        space = SearchSpace(
            kernel=["rbf", "linear"],
            gamma=(1e-4, 10.0, "log"),
        )
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")

        adapter = GFOSearchSpaceAdapter(space)
        constraints = adapter.get_constraints()

        # No constraints should be returned (conditions are NOT constraints)
        assert constraints is None

    def test_gfo_constraints_only_not_conditions(self):
        """Test GFO returns only explicit constraints, not conditions."""
        space = SearchSpace(
            kernel=["rbf", "linear"],
            C=(0.01, 100.0, "log"),
            gamma=(1e-4, 10.0, "log"),
        )
        # Add a condition (should NOT become a constraint)
        space.add_condition("gamma", when=lambda p: p["kernel"] == "rbf")
        # Add an explicit constraint (SHOULD be returned)
        space.add_constraint(lambda p: p["C"] < 50)

        adapter = GFOSearchSpaceAdapter(space)
        constraints = adapter.get_constraints()

        # Should have exactly 1 constraint (the explicit one)
        assert constraints is not None
        assert len(constraints) == 1

        # Verify it's the right constraint
        assert constraints[0]({"C": 10}) is True
        assert constraints[0]({"C": 60}) is False

    def test_gfo_multiple_conditions_no_constraints(self):
        """Test multiple conditions don't create any constraints."""
        space = SearchSpace(
            model=["svm", "rf", "nn"],
            C=(0.01, 100.0, "log"),
            n_estimators=[10, 50, 100],
            hidden_size=[32, 64, 128],
        )
        space.add_condition("C", when=lambda p: p["model"] == "svm")
        space.add_condition("n_estimators", when=lambda p: p["model"] == "rf")
        space.add_condition("hidden_size", when=lambda p: p["model"] == "nn")

        adapter = GFOSearchSpaceAdapter(space)
        constraints = adapter.get_constraints()

        # No constraints - conditions are not constraints
        assert constraints is None

    def test_gfo_constraint_function_works(self):
        """Test extracted constraint functions work correctly."""
        space = SearchSpace(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5])
        space.add_constraint(lambda p: p["x"] + p["y"] < 6)

        adapter = GFOSearchSpaceAdapter(space)
        constraints = adapter.get_constraints()

        # Test the constraint function
        assert constraints[0]({"x": 1, "y": 1}) is True  # 2 < 6
        assert constraints[0]({"x": 2, "y": 2}) is True  # 4 < 6
        assert constraints[0]({"x": 3, "y": 3}) is False  # 6 not < 6
        assert constraints[0]({"x": 5, "y": 5}) is False  # 10 not < 6

    def test_sklearn_filter_invalid(self):
        """Test sklearn adapter filters invalid combinations."""
        space = SearchSpace(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5])
        space.add_constraint(lambda p: p["x"] + p["y"] <= 4)

        adapter = SklearnSearchSpaceAdapter(space)

        # Generate all combinations
        from itertools import product

        all_combos = [{"x": x, "y": y} for x, y in product([1, 2, 3, 4, 5], repeat=2)]

        # Filter
        valid = adapter.filter_invalid_combinations(all_combos)

        # Check all valid combinations satisfy constraint
        for combo in valid:
            assert combo["x"] + combo["y"] <= 4


class TestOptunaAdapterAdvanced:
    """Tests for advanced Optuna adapter functionality."""

    @pytest.fixture
    def skip_if_no_optuna(self):
        """Skip test if optuna is not available."""
        pytest.importorskip("optuna")

    def test_log_scale_creates_float_distribution(self, skip_if_no_optuna):
        """Test that log-scale creates FloatDistribution."""
        import optuna.distributions

        space = SearchSpace(lr=(1e-5, 1e-1, "log"))
        adapted = space.to_backend("optuna")

        # This correctly creates a FloatDistribution
        assert isinstance(adapted["lr"], optuna.distributions.FloatDistribution)
        assert adapted["lr"].log is True

    def test_suggest_params_handles_float_distribution(self, skip_if_no_optuna):
        """Test _suggest_params correctly handles FloatDistribution.

        The _suggest_params method in _base_optuna_adapter.py checks for
        optuna.distributions.BaseDistribution instances and uses trial._suggest.
        """
        import optuna
        import optuna.distributions

        from hyperactive.opt._adapters._base_optuna_adapter import _BaseOptunaAdapter

        # Create a mock adapter to test _suggest_params directly
        class MockOptunaAdapter(_BaseOptunaAdapter):
            def _get_optimizer(self):
                return optuna.samplers.TPESampler()

        adapter = MockOptunaAdapter(
            param_space={"x": [1, 2, 3]},
            n_trials=1,
            experiment=lambda p: 1.0,
        )

        # Create a param_space with FloatDistribution (as created by SearchSpace)
        param_space = {
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
        }

        # Create a trial
        study = optuna.create_study()
        trial = study.ask()

        params = adapter._suggest_params(trial, param_space)
        assert "lr" in params
        assert 1e-5 <= params["lr"] <= 1e-1

    def test_integer_continuous_returns_int(self, skip_if_no_optuna):
        """Test integer continuous returns int.

        When SearchSpace has (1, 10) with ints, the Optuna adapter
        uses suggest_int to return an integer.
        """
        import optuna

        from hyperactive.opt._adapters._base_optuna_adapter import _BaseOptunaAdapter

        class MockOptunaAdapter(_BaseOptunaAdapter):
            def _get_optimizer(self):
                return optuna.samplers.TPESampler()

        adapter = MockOptunaAdapter(
            param_space={"x": [1, 2, 3]},
            n_trials=1,
            experiment=lambda p: 1.0,
        )

        # Integer range as tuple
        param_space = {"n_layers": (1, 10)}

        study = optuna.create_study()
        trial = study.ask()

        params = adapter._suggest_params(trial, param_space)
        assert "n_layers" in params
        # Should be an integer
        assert isinstance(params["n_layers"], int)


class TestAdapterEdgeCases:
    """Tests for edge cases in adapters."""

    def test_gfo_with_all_dimension_types(self):
        """Test GFO adapter handles all dimension types together."""
        import scipy.stats as st

        space = SearchSpace(
            categorical=["a", "b", "c"],
            discrete=np.arange(10),
            continuous=(0.0, 10.0),
            continuous_int=(1, 100),
            log_scale=(1e-5, 1e-1, "log"),
            distribution=st.uniform(0, 1),
            constant=42,
        )

        gfo_space = space.to_backend("gfo", resolution=20)

        assert len(gfo_space["categorical"]) == 3
        assert len(gfo_space["discrete"]) == 10
        assert len(gfo_space["continuous"]) == 20
        assert len(gfo_space["continuous_int"]) == 100  # All integers in range
        assert len(gfo_space["log_scale"]) == 20
        assert len(gfo_space["constant"]) == 1
        assert gfo_space["constant"][0] == 42

    def test_sklearn_with_all_dimension_types(self):
        """Test sklearn adapter handles all dimension types."""
        import scipy.stats as st

        space = SearchSpace(
            categorical=["a", "b", "c"],
            discrete=np.arange(5),
            continuous=(0.0, 10.0),
            log_scale=(1e-5, 1e-1, "log"),
            distribution=st.uniform(0, 1),
            constant=42,
        )

        sklearn_space = space.to_backend("sklearn_random")

        assert sklearn_space["categorical"] == ["a", "b", "c"]
        assert sklearn_space["discrete"] == [0, 1, 2, 3, 4]
        assert hasattr(sklearn_space["continuous"], "rvs")  # scipy distribution
        assert hasattr(sklearn_space["log_scale"], "rvs")
        assert sklearn_space["distribution"] is not None
        assert sklearn_space["constant"] == [42]

    def test_empty_search_space(self):
        """Test adapters handle empty search space."""
        space = SearchSpace()

        gfo_space = space.to_backend("gfo")
        assert gfo_space == {}

        sklearn_space = space.to_backend("sklearn_random")
        assert sklearn_space == {}

    def test_single_value_categorical(self):
        """Test adapters handle single-value categorical."""
        space = SearchSpace(x=["only_option"])

        gfo_space = space.to_backend("gfo")
        assert len(gfo_space["x"]) == 1
        assert gfo_space["x"][0] == "only_option"
