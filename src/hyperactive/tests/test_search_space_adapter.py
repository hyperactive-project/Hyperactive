"""Tests for SearchSpaceAdapter encoding/decoding."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest

from hyperactive.opt._adapters._search_space_adapter import SearchSpaceAdapter


class TestSearchSpaceAdapter:
    """Tests for SearchSpaceAdapter encoding/decoding."""

    def test_encode_categorical_to_integers(self):
        """Categorical strings are encoded to integer indices."""
        space = {"kernel": ["rbf", "linear", "poly"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1, 2]}
        assert adapter.categorical_mapping == {
            "kernel": {0: "rbf", 1: "linear", 2: "poly"}
        }

    def test_decode_integers_to_categorical(self):
        """Integer indices are decoded back to original strings."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": 1})

        assert decoded == {"kernel": "linear"}

    def test_no_encoding_when_supported(self):
        """No encoding when backend supports categorical."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": True})

        assert adapter.needs_encoding is False
        assert adapter.encode() is space  # Same object, not copied

    def test_mixed_dimensions(self):
        """Categorical and numeric dimensions coexist."""
        space = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1], "C": [0.1, 1, 10]}
        assert "kernel" in adapter.categorical_mapping
        assert "C" not in adapter.categorical_mapping

    def test_wrapped_experiment_decodes(self):
        """Wrapped experiment receives decoded params."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        received_params = []

        class MockExperiment:
            def score(self, params):
                received_params.append(params.copy())
                return 1.0

        wrapped = adapter.wrap_experiment(MockExperiment())
        wrapped.score({"kernel": 1})

        assert received_params[0] == {"kernel": "linear"}

    def test_numpy_float_handling(self):
        """Numpy float indices are converted correctly."""
        np = pytest.importorskip("numpy")

        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": np.float64(0.0)})

        assert decoded == {"kernel": "rbf"}

    def test_numpy_int_handling(self):
        """Numpy integer indices are converted correctly."""
        np = pytest.importorskip("numpy")

        space = {"kernel": ["rbf", "linear", "poly"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": np.int64(2)})

        assert decoded == {"kernel": "poly"}

    def test_no_encoding_for_numeric_only_space(self):
        """No encoding needed when space contains only numeric values."""
        space = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        assert adapter.needs_encoding is False
        assert adapter.encode() is space

    def test_wrapped_experiment_callable(self):
        """Wrapped experiment is callable like original."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        class MockExperiment:
            def score(self, params):
                return 1.0 if params["kernel"] == "linear" else 0.5

        wrapped = adapter.wrap_experiment(MockExperiment())

        # Call via __call__
        result = wrapped({"kernel": 1})
        assert result == 1.0

    def test_wrapped_experiment_evaluate(self):
        """Wrapped experiment.evaluate() also decodes."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        received_params = []

        class MockExperiment:
            def score(self, params):
                return 1.0

            def evaluate(self, params):
                received_params.append(params.copy())
                return {"accuracy": 0.95}

        wrapped = adapter.wrap_experiment(MockExperiment())
        wrapped.evaluate({"kernel": 0})

        assert received_params[0] == {"kernel": "rbf"}

    def test_wrapped_experiment_forwards_attributes(self):
        """Wrapped experiment forwards attribute access to original."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        class MockExperiment:
            custom_attr = "test_value"

            def score(self, params):
                return 1.0

        wrapped = adapter.wrap_experiment(MockExperiment())

        assert wrapped.custom_attr == "test_value"

    def test_decode_preserves_non_categorical_params(self):
        """Decode preserves parameters that weren't encoded."""
        space = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10]}
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})
        adapter.encode()

        decoded = adapter.decode({"kernel": 1, "C": 1, "extra": "value"})

        assert decoded == {"kernel": "linear", "C": 1, "extra": "value"}

    def test_default_capability_is_categorical_supported(self):
        """Default capability assumes categorical is supported."""
        space = {"kernel": ["rbf", "linear"]}
        adapter = SearchSpaceAdapter(space, capabilities={})

        assert adapter.needs_encoding is False

    def test_multiple_categorical_dimensions(self):
        """Multiple categorical dimensions are all encoded."""
        space = {
            "kernel": ["rbf", "linear"],
            "solver": ["lbfgs", "sgd", "adam"],
        }
        adapter = SearchSpaceAdapter(space, capabilities={"categorical": False})

        encoded = adapter.encode()

        assert encoded == {"kernel": [0, 1], "solver": [0, 1, 2]}
        assert adapter.categorical_mapping["kernel"] == {0: "rbf", 1: "linear"}
        assert adapter.categorical_mapping["solver"] == {0: "lbfgs", 1: "sgd", 2: "adam"}

        decoded = adapter.decode({"kernel": 0, "solver": 2})
        assert decoded == {"kernel": "rbf", "solver": "adam"}


class TestCapabilityTags:
    """Tests for capability tags related to categorical encoding."""

    def test_cmaes_has_categorical_false_tag(self):
        """CMA-ES optimizer should have categorical capability set to False."""
        from hyperactive.opt.optuna import CmaEsOptimizer

        opt = CmaEsOptimizer.create_test_instance()
        assert opt.get_tag("capability:categorical") is False

    def test_optuna_optimizers_have_categorical_true_tag(self):
        """Optuna TPE/Random/Grid optimizers should support categorical."""
        from hyperactive.opt.optuna import GridOptimizer, RandomOptimizer, TPEOptimizer

        for OptCls in [TPEOptimizer, RandomOptimizer, GridOptimizer]:
            opt = OptCls.create_test_instance()
            assert opt.get_tag("capability:categorical") is True

    def test_gfo_optimizers_have_categorical_false_tag(self):
        """GFO optimizers should have categorical tag set to False."""
        from hyperactive.opt.gfo import RandomSearch

        opt = RandomSearch.create_test_instance()
        # GFO does not support categorical natively - adapter handles encoding
        assert opt.get_tag("capability:categorical") is False
