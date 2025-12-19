"""Tests for the meta_learning module."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

# Suppress warnings during tests
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"


# =============================================================================
# Tests for BenchmarkConfig and BenchmarkResult
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_create_config(self):
        """BenchmarkConfig should be creatable with required fields."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkConfig
        from hyperactive.experiment.bench import Sphere

        config = BenchmarkConfig(
            name="test_config",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [1, 2, 3], "x1": [1, 2, 3]},
            n_iter=10,
        )

        assert config.name == "test_config"
        assert config.experiment_class == Sphere
        assert config.experiment_params == {"n_dim": 2}
        assert config.n_iter == 10

    def test_config_stores_search_space(self):
        """BenchmarkConfig should store search space correctly."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkConfig
        from hyperactive.experiment.bench import Ackley

        search_space = {"x": list(range(10)), "y": list(range(10))}
        config = BenchmarkConfig(
            name="ackley_test",
            experiment_class=Ackley,
            experiment_params={"d": 2},
            search_space=search_space,
            n_iter=50,
        )

        assert config.search_space == search_space
        assert len(config.search_space) == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self):
        """BenchmarkResult should be creatable with required fields."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkResult

        result = BenchmarkResult(
            config_name="test_config",
            optimizer_name="RandomSearch",
            best_score=-1.5,
            n_iter=100,
            random_state=42,
        )

        assert result.config_name == "test_config"
        assert result.optimizer_name == "RandomSearch"
        assert result.best_score == -1.5
        assert result.n_iter == 100
        assert result.random_state == 42

    def test_result_default_extra(self):
        """BenchmarkResult should have empty dict as default extra."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkResult

        result = BenchmarkResult(
            config_name="test",
            optimizer_name="HillClimbing",
            best_score=0.0,
            n_iter=10,
            random_state=0,
        )

        assert result.extra == {}

    def test_result_with_extra(self):
        """BenchmarkResult should accept extra dict."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkResult

        result = BenchmarkResult(
            config_name="test",
            optimizer_name="HillClimbing",
            best_score=0.0,
            n_iter=10,
            random_state=0,
            extra={"runtime": 1.5},
        )

        assert result.extra == {"runtime": 1.5}


# =============================================================================
# Tests for BenchmarkDataCollector
# =============================================================================


class TestBenchmarkDataCollectorInit:
    """Tests for BenchmarkDataCollector initialization."""

    def test_default_init(self):
        """BenchmarkDataCollector should initialize with defaults."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector

        collector = BenchmarkDataCollector()

        assert collector.n_runs == 3
        assert collector.random_seed_base == 42
        assert collector.verbose is False

    def test_custom_init(self):
        """BenchmarkDataCollector should accept custom parameters."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector

        collector = BenchmarkDataCollector(
            n_runs=5,
            random_seed_base=100,
            verbose=True,
        )

        assert collector.n_runs == 5
        assert collector.random_seed_base == 100
        assert collector.verbose is True


class TestBenchmarkDataCollectorConfigs:
    """Tests for BenchmarkDataCollector configuration methods."""

    def test_get_default_configs_returns_list(self):
        """get_default_configs should return a list."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector

        collector = BenchmarkDataCollector()
        configs = collector.get_default_configs()

        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_default_configs_have_required_fields(self):
        """Default configs should have all required fields."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector

        collector = BenchmarkDataCollector()
        configs = collector.get_default_configs()

        for config in configs:
            assert hasattr(config, "name")
            assert hasattr(config, "experiment_class")
            assert hasattr(config, "experiment_params")
            assert hasattr(config, "search_space")
            assert hasattr(config, "n_iter")

    def test_default_configs_include_different_benchmarks(self):
        """Default configs should include Sphere, Ackley, and Parabola."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector

        collector = BenchmarkDataCollector()
        configs = collector.get_default_configs()

        names = [c.name for c in configs]
        assert any("sphere" in n for n in names)
        assert any("ackley" in n for n in names)
        assert any("parabola" in n for n in names)

    def test_get_default_optimizers_returns_list(self):
        """get_default_optimizers should return a list of optimizer classes."""
        from hyperactive.algorithm_selection.meta_learning import BenchmarkDataCollector
        from hyperactive.base import BaseOptimizer

        collector = BenchmarkDataCollector()
        optimizers = collector.get_default_optimizers()

        assert isinstance(optimizers, list)
        assert len(optimizers) > 0
        for opt in optimizers:
            assert isinstance(opt, type)
            assert issubclass(opt, BaseOptimizer)


class TestBenchmarkDataCollectorCollect:
    """Tests for BenchmarkDataCollector.collect() method."""

    def test_collect_returns_list(self):
        """collect() should return a list of BenchmarkResult."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            BenchmarkResult,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="test_sphere",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1, 2], "x1": [0, 1, 2]},
            n_iter=5,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect([config], [RandomSearch])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)

    def test_collect_multiple_configs_and_optimizers(self):
        """collect() should run all combinations."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch, HillClimbing

        configs = [
            BenchmarkConfig(
                name="config1",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": [0, 1], "x1": [0, 1]},
                n_iter=3,
            ),
            BenchmarkConfig(
                name="config2",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": [0, 1, 2], "x1": [0, 1, 2]},
                n_iter=3,
            ),
        ]

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect(configs, [RandomSearch, HillClimbing])

        # 2 configs × 2 optimizers × 1 run = 4 results
        assert len(results) == 4

    def test_collect_respects_n_runs(self):
        """collect() should run n_runs times per combination."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        collector = BenchmarkDataCollector(n_runs=3, verbose=False)
        results = collector.collect([config], [RandomSearch])

        # 1 config × 1 optimizer × 3 runs = 3 results
        assert len(results) == 3

    def test_collect_results_have_correct_fields(self):
        """Results should have correct config and optimizer names."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="my_test_config",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect([config], [RandomSearch])

        assert results[0].config_name == "my_test_config"
        assert results[0].optimizer_name == "RandomSearch"
        assert results[0].n_iter == 3

    def test_collect_to_dataframe(self):
        """collect_to_dataframe() should return pandas DataFrame."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        df = collector.collect_to_dataframe([config], [RandomSearch])

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert "config_name" in df.columns
        assert "optimizer_name" in df.columns
        assert "best_score" in df.columns


# =============================================================================
# Tests for DatasetGenerator
# =============================================================================


class TestDatasetGeneratorInit:
    """Tests for DatasetGenerator initialization."""

    def test_default_init(self):
        """DatasetGenerator should initialize with default aggregation."""
        from hyperactive.algorithm_selection.meta_learning import DatasetGenerator

        generator = DatasetGenerator()
        assert generator.aggregation == "mean"

    def test_custom_aggregation(self):
        """DatasetGenerator should accept custom aggregation."""
        from hyperactive.algorithm_selection.meta_learning import DatasetGenerator

        for agg in ["mean", "median", "best"]:
            generator = DatasetGenerator(aggregation=agg)
            assert generator.aggregation == agg

    def test_invalid_aggregation_raises(self):
        """DatasetGenerator should raise for invalid aggregation."""
        from hyperactive.algorithm_selection.meta_learning import DatasetGenerator

        with pytest.raises(ValueError):
            DatasetGenerator(aggregation="invalid")


class TestDatasetGeneratorGenerate:
    """Tests for DatasetGenerator.generate() method."""

    def test_generate_returns_correct_types(self):
        """generate() should return X, y, feature_names, optimizer_names."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            DatasetGenerator,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch, HillClimbing

        config = BenchmarkConfig(
            name="sphere_test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1, 2], "x1": [0, 1, 2]},
            n_iter=5,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect([config], [RandomSearch, HillClimbing])

        generator = DatasetGenerator()
        X, y, feature_names, optimizer_names = generator.generate(results, [config])

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert isinstance(optimizer_names, list)

    def test_generate_correct_shapes(self):
        """generate() should return arrays with correct shapes."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            DatasetGenerator,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch, HillClimbing

        configs = [
            BenchmarkConfig(
                name="config1",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": [0, 1], "x1": [0, 1]},
                n_iter=3,
            ),
            BenchmarkConfig(
                name="config2",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": [0, 1, 2], "x1": [0, 1, 2]},
                n_iter=3,
            ),
        ]

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect(configs, [RandomSearch, HillClimbing])

        generator = DatasetGenerator()
        X, y, feature_names, optimizer_names = generator.generate(results, configs)

        # 2 configs = 2 examples
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        # Features should match feature_names
        assert X.shape[1] == len(feature_names)

    def test_generate_feature_names(self):
        """generate() should return expected feature names."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            DatasetGenerator,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="sphere_test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect([config], [RandomSearch])

        generator = DatasetGenerator()
        _, _, feature_names, _ = generator.generate(results, [config])

        assert "n_dimensions" in feature_names
        assert "total_size" in feature_names
        assert "n_iter" in feature_names

    def test_generate_empty_results_raises(self):
        """generate() should raise ValueError for empty results."""
        from hyperactive.algorithm_selection.meta_learning import DatasetGenerator

        generator = DatasetGenerator()

        with pytest.raises(ValueError, match="No valid examples"):
            generator.generate([], [])


class TestDatasetGeneratorAggregation:
    """Tests for DatasetGenerator aggregation methods."""

    def test_mean_aggregation(self):
        """Mean aggregation should average scores."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkResult,
            DatasetGenerator,
            BenchmarkConfig,
        )
        from hyperactive.experiment.bench import Sphere

        config = BenchmarkConfig(
            name="test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        # Create results with known scores
        results = [
            BenchmarkResult("test", "OptA", 10.0, 3, 0),
            BenchmarkResult("test", "OptA", 20.0, 3, 1),
            BenchmarkResult("test", "OptB", 5.0, 3, 0),
        ]

        generator = DatasetGenerator(aggregation="mean")
        aggregated = generator._aggregate_scores(results)

        assert aggregated["OptA"] == 15.0  # (10 + 20) / 2
        assert aggregated["OptB"] == 5.0

    def test_best_aggregation(self):
        """Best aggregation should take max score."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkResult,
            DatasetGenerator,
        )

        results = [
            BenchmarkResult("test", "OptA", 10.0, 3, 0),
            BenchmarkResult("test", "OptA", 20.0, 3, 1),
        ]

        generator = DatasetGenerator(aggregation="best")
        aggregated = generator._aggregate_scores(results)

        assert aggregated["OptA"] == 20.0


# =============================================================================
# Tests for AlgorithmSelectionModel
# =============================================================================


class TestAlgorithmSelectionModelInit:
    """Tests for AlgorithmSelectionModel initialization."""

    def test_default_init(self):
        """AlgorithmSelectionModel should initialize with defaults."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        model = AlgorithmSelectionModel()

        assert model.model_type == "random_forest"
        assert model.is_fitted_ is False
        assert model.model_ is None

    def test_gradient_boosting_init(self):
        """AlgorithmSelectionModel should accept gradient_boosting type."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        model = AlgorithmSelectionModel(model_type="gradient_boosting")
        assert model.model_type == "gradient_boosting"

    def test_invalid_model_type_raises(self):
        """AlgorithmSelectionModel should raise for invalid model type."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        with pytest.raises(ValueError):
            AlgorithmSelectionModel(model_type="invalid_type")


class TestAlgorithmSelectionModelFit:
    """Tests for AlgorithmSelectionModel.fit() method."""

    def test_fit_sets_is_fitted(self):
        """fit() should set is_fitted_ to True."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        assert model.is_fitted_ is True

    def test_fit_stores_names(self):
        """fit() should store feature and optimizer names."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["feature1", "feature2"], ["OptA", "OptB"])

        assert model.feature_names_ == ["feature1", "feature2"]
        assert model.optimizer_names_ == ["OptA", "OptB"]

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        result = model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        assert result is model


class TestAlgorithmSelectionModelPredict:
    """Tests for AlgorithmSelectionModel prediction methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])
        return model

    def test_predict_returns_names(self, fitted_model):
        """predict() should return optimizer names."""
        X_test = np.array([[2, 3]])
        predictions = fitted_model.predict(X_test)

        assert isinstance(predictions, list)
        assert predictions[0] in ["OptA", "OptB"]

    def test_predict_proba_returns_dict(self, fitted_model):
        """predict_proba() should return dict of probabilities."""
        X_test = np.array([[2, 3]])
        proba = fitted_model.predict_proba(X_test)

        assert isinstance(proba, dict)
        assert "OptA" in proba
        assert "OptB" in proba

    def test_rank_returns_sorted_dict(self, fitted_model):
        """rank() should return sorted dict of scores."""
        features = np.array([2, 3])
        rankings = fitted_model.rank(features)

        assert isinstance(rankings, dict)
        # Should be sorted descending
        scores = list(rankings.values())
        assert scores == sorted(scores, reverse=True)

    def test_predict_unfitted_raises(self):
        """predict() on unfitted model should raise."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        model = AlgorithmSelectionModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(np.array([[1, 2]]))


class TestAlgorithmSelectionModelSaveLoad:
    """Tests for AlgorithmSelectionModel save/load methods."""

    def test_save_creates_file(self):
        """save() should create a file."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            model.save(path)
            assert path.exists()
        finally:
            path.unlink()

    def test_load_restores_model(self):
        """load() should restore a saved model."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            model.save(path)
            loaded = AlgorithmSelectionModel.load(path)

            assert loaded.is_fitted_
            assert loaded.feature_names_ == ["f1", "f2"]
            assert loaded.optimizer_names_ == ["OptA", "OptB"]
        finally:
            path.unlink()

    def test_load_nonexistent_raises(self):
        """load() should raise for nonexistent file."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        with pytest.raises(FileNotFoundError):
            AlgorithmSelectionModel.load("/nonexistent/path.pkl")


class TestAlgorithmSelectionModelEvaluate:
    """Tests for AlgorithmSelectionModel.evaluate() method."""

    def test_evaluate_returns_metrics(self):
        """evaluate() should return dict with accuracy."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        metrics = model.evaluate(X, y)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestAlgorithmSelectionModelFeatureImportance:
    """Tests for feature importance."""

    def test_get_feature_importance(self):
        """get_feature_importance() should return dict."""
        from hyperactive.algorithm_selection.meta_learning import AlgorithmSelectionModel

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])

        model = AlgorithmSelectionModel()
        model.fit(X, y, ["f1", "f2"], ["OptA", "OptB"])

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert "f1" in importance
        assert "f2" in importance


# =============================================================================
# Tests for OptimizerRankingModel
# =============================================================================


class TestOptimizerRankingModel:
    """Tests for OptimizerRankingModel class."""

    def test_is_model_available(self):
        """is_model_available() should return bool."""
        from hyperactive.algorithm_selection.meta_learning import OptimizerRankingModel

        result = OptimizerRankingModel.is_model_available()
        assert isinstance(result, bool)

    def test_get_model_path(self):
        """get_model_path() should return Path."""
        from hyperactive.algorithm_selection.meta_learning import OptimizerRankingModel

        path = OptimizerRankingModel.get_model_path()
        assert isinstance(path, Path)

    def test_clear_cache(self):
        """clear_cache() should clear the cached model."""
        from hyperactive.algorithm_selection.meta_learning import OptimizerRankingModel

        # Should not raise
        OptimizerRankingModel.clear_cache()
        assert OptimizerRankingModel._model is None

    def test_rank_optimizers_with_dict(self):
        """rank_optimizers() should work with ss_features_dict."""
        from hyperactive.algorithm_selection.meta_learning import OptimizerRankingModel

        # Only test if model is available
        if not OptimizerRankingModel.is_model_available():
            pytest.skip("Pre-trained model not available")

        OptimizerRankingModel.clear_cache()

        ss_features = {
            "n_dimensions": 2,
            "total_size": 100,
            "n_continuous": 0,
            "n_discrete": 2,
            "n_categorical": 0,
            "avg_choices_per_dim": 10,
            "min_choices": 10,
            "max_choices": 10,
            "avg_range_span": 10.0,
            "has_mixed_types": False,
        }

        rankings = OptimizerRankingModel.rank_optimizers(
            problem_features=[],
            ss_features_dict=ss_features,
            n_iter=100,
        )

        assert isinstance(rankings, dict)
        if rankings:  # May be empty if model loading fails
            assert all(isinstance(k, str) for k in rankings.keys())
            assert all(isinstance(v, float) for v in rankings.values())


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetaLearningIntegration:
    """Integration tests for the full meta-learning pipeline."""

    def test_full_pipeline_small(self):
        """Test full pipeline with minimal data."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            DatasetGenerator,
            AlgorithmSelectionModel,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch, HillClimbing

        # Create minimal configs
        configs = [
            BenchmarkConfig(
                name="sphere_small",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": [0, 1, 2], "x1": [0, 1, 2]},
                n_iter=5,
            ),
            BenchmarkConfig(
                name="sphere_large",
                experiment_class=Sphere,
                experiment_params={"n_dim": 2},
                search_space={"x0": list(range(10)), "x1": list(range(10))},
                n_iter=5,
            ),
        ]

        # Collect
        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect(configs, [RandomSearch, HillClimbing])
        assert len(results) == 4

        # Generate dataset
        generator = DatasetGenerator()
        X, y, feature_names, optimizer_names = generator.generate(results, configs)
        assert X.shape[0] == 2

        # Train
        model = AlgorithmSelectionModel()
        model.fit(X, y, feature_names, optimizer_names)
        assert model.is_fitted_

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == 2

    def test_pipeline_with_save_load(self):
        """Test pipeline including save and load."""
        from hyperactive.algorithm_selection.meta_learning import (
            BenchmarkDataCollector,
            BenchmarkConfig,
            DatasetGenerator,
            AlgorithmSelectionModel,
        )
        from hyperactive.experiment.bench import Sphere
        from hyperactive.opt import RandomSearch

        config = BenchmarkConfig(
            name="test",
            experiment_class=Sphere,
            experiment_params={"n_dim": 2},
            search_space={"x0": [0, 1], "x1": [0, 1]},
            n_iter=3,
        )

        collector = BenchmarkDataCollector(n_runs=1, verbose=False)
        results = collector.collect([config], [RandomSearch])

        generator = DatasetGenerator()
        X, y, feature_names, optimizer_names = generator.generate(results, [config])

        model = AlgorithmSelectionModel()
        model.fit(X, y, feature_names, optimizer_names)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            model.save(path)
            loaded = AlgorithmSelectionModel.load(path)

            # Loaded model should give same predictions
            orig_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            assert orig_pred == loaded_pred
        finally:
            path.unlink()
