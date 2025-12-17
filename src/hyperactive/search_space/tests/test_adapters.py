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
        assert len(constraints) >= 1

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
