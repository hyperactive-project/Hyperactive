"""API surface smoke tests for Hyperactive.

Purely structural: verifies import paths and method presence on classes.
No instantiation, no computation, no I/O. If any documented import path
breaks or a public method disappears, these tests fail immediately.

The hardcoded name lists below ARE the API specification. Removing an entry
means the public surface intentionally shrank; adding a class to the package
without updating these lists will also fail (via registry completeness checks).
"""

import pytest

GFO_OPTIMIZER_NAMES = [
    "BayesianOptimizer",
    "DifferentialEvolution",
    "DirectAlgorithm",
    "DownhillSimplexOptimizer",
    "EvolutionStrategy",
    "ForestOptimizer",
    "GeneticAlgorithm",
    "GridSearch",
    "HillClimbing",
    "LipschitzOptimizer",
    "ParallelTempering",
    "ParticleSwarmOptimizer",
    "PatternSearch",
    "PowellsMethod",
    "RandomRestartHillClimbing",
    "RandomSearch",
    "RepulsingHillClimbing",
    "SimulatedAnnealing",
    "SpiralOptimization",
    "StochasticHillClimbing",
    "TreeStructuredParzenEstimators",
]

OPTUNA_OPTIMIZER_NAMES = [
    "CmaEsOptimizer",
    "GPOptimizer",
    "GridOptimizer",
    "NSGAIIOptimizer",
    "NSGAIIIOptimizer",
    "QMCOptimizer",
    "RandomOptimizer",
    "TPEOptimizer",
]

SK_OPTIMIZER_NAMES = ["GridSearchSk", "RandomSearchSk"]

BENCHMARK_NAMES = ["Ackley", "Parabola", "Sphere"]

EXPERIMENT_INTEGRATION_NAMES = [
    "LightGBMExperiment",
    "SklearnCvExperiment",
    "SkproProbaRegExperiment",
    "SktimeClassificationExperiment",
    "SktimeForecastingExperiment",
    "TorchExperiment",
]

OPTIMIZER_METHODS = [
    "clone",
    "get_experiment",
    "get_params",
    "get_search_config",
    "get_tag",
    "get_tags",
    "get_test_params",
    "set_params",
    "set_tags",
    "solve",
]

EXPERIMENT_METHODS = [
    "__call__",
    "clone",
    "evaluate",
    "get_params",
    "get_tag",
    "get_tags",
    "get_test_params",
    "paramnames",
    "score",
    "set_params",
    "set_tags",
]


class TestTopLevelAttributes:
    """Package-level attributes are accessible."""

    def test_version_is_string(self):
        import hyperactive

        assert isinstance(hyperactive.__version__, str)

    def test_license_value(self):
        import hyperactive

        assert hyperactive.__license__ == "MIT"


class TestBaseClassImports:
    """Base classes importable with full method surface."""

    def test_base_optimizer_from_base(self):
        from hyperactive.base import BaseOptimizer

        assert BaseOptimizer is not None

    def test_base_experiment_from_base(self):
        from hyperactive.base import BaseExperiment

        assert BaseExperiment is not None

    def test_base_experiment_from_experiment(self):
        from hyperactive.experiment import BaseExperiment

        assert BaseExperiment is not None

    def test_base_experiment_same_class_both_paths(self):
        from hyperactive.base import BaseExperiment as A
        from hyperactive.experiment import BaseExperiment as B

        assert A is B

    @pytest.mark.parametrize("method", OPTIMIZER_METHODS)
    def test_base_optimizer_has_method(self, method):
        from hyperactive.base import BaseOptimizer

        assert hasattr(BaseOptimizer, method)

    @pytest.mark.parametrize("method", EXPERIMENT_METHODS)
    def test_base_experiment_has_method(self, method):
        from hyperactive.base import BaseExperiment

        assert hasattr(BaseExperiment, method)


class TestGfoOptimizerImports:
    """All GFO optimizers importable from hyperactive.opt and hyperactive.opt.gfo."""

    @pytest.mark.parametrize("name", GFO_OPTIMIZER_NAMES)
    def test_importable_from_opt(self, name):
        import hyperactive.opt as mod

        assert hasattr(mod, name), f"{name} not in hyperactive.opt"

    @pytest.mark.parametrize("name", GFO_OPTIMIZER_NAMES)
    def test_importable_from_opt_gfo(self, name):
        import hyperactive.opt.gfo as mod

        assert hasattr(mod, name), f"{name} not in hyperactive.opt.gfo"

    @pytest.mark.parametrize("name", GFO_OPTIMIZER_NAMES)
    def test_is_base_optimizer_subclass(self, name):
        import hyperactive.opt.gfo as mod
        from hyperactive.base import BaseOptimizer

        cls = getattr(mod, name)
        assert issubclass(cls, BaseOptimizer)

    @pytest.mark.parametrize("name", GFO_OPTIMIZER_NAMES)
    def test_has_optimizer_methods(self, name):
        import hyperactive.opt.gfo as mod

        cls = getattr(mod, name)
        for method in OPTIMIZER_METHODS:
            assert hasattr(cls, method), f"{name} missing {method}"

    @pytest.mark.parametrize("name", GFO_OPTIMIZER_NAMES)
    def test_same_class_both_paths(self, name):
        import hyperactive.opt as opt_mod
        import hyperactive.opt.gfo as gfo_mod

        assert getattr(opt_mod, name) is getattr(gfo_mod, name)


class TestOptunaOptimizerImports:
    """All Optuna optimizers importable from hyperactive.opt and .opt.optuna."""

    @pytest.mark.parametrize("name", OPTUNA_OPTIMIZER_NAMES)
    def test_importable_from_opt(self, name):
        import hyperactive.opt as mod

        assert hasattr(mod, name), f"{name} not in hyperactive.opt"

    @pytest.mark.parametrize("name", OPTUNA_OPTIMIZER_NAMES)
    def test_importable_from_opt_optuna(self, name):
        import hyperactive.opt.optuna as mod

        assert hasattr(mod, name), f"{name} not in hyperactive.opt.optuna"

    @pytest.mark.parametrize("name", OPTUNA_OPTIMIZER_NAMES)
    def test_is_base_optimizer_subclass(self, name):
        import hyperactive.opt.optuna as mod
        from hyperactive.base import BaseOptimizer

        cls = getattr(mod, name)
        assert issubclass(cls, BaseOptimizer)

    @pytest.mark.parametrize("name", OPTUNA_OPTIMIZER_NAMES)
    def test_has_optimizer_methods(self, name):
        import hyperactive.opt.optuna as mod

        cls = getattr(mod, name)
        for method in OPTIMIZER_METHODS:
            assert hasattr(cls, method), f"{name} missing {method}"

    @pytest.mark.parametrize("name", OPTUNA_OPTIMIZER_NAMES)
    def test_same_class_both_paths(self, name):
        import hyperactive.opt as opt_mod
        import hyperactive.opt.optuna as optuna_mod

        assert getattr(opt_mod, name) is getattr(optuna_mod, name)


class TestSkStyleOptimizerImports:
    """GridSearchSk and RandomSearchSk importable with full optimizer interface."""

    @pytest.mark.parametrize("name", SK_OPTIMIZER_NAMES)
    def test_importable_from_opt(self, name):
        import hyperactive.opt as mod

        assert hasattr(mod, name), f"{name} not in hyperactive.opt"

    @pytest.mark.parametrize("name", SK_OPTIMIZER_NAMES)
    def test_is_base_optimizer_subclass(self, name):
        import hyperactive.opt as mod
        from hyperactive.base import BaseOptimizer

        cls = getattr(mod, name)
        assert issubclass(cls, BaseOptimizer)

    @pytest.mark.parametrize("name", SK_OPTIMIZER_NAMES)
    def test_has_optimizer_methods(self, name):
        import hyperactive.opt as mod

        cls = getattr(mod, name)
        for method in OPTIMIZER_METHODS:
            assert hasattr(cls, method), f"{name} missing {method}"


class TestFunctionExperimentImport:
    """FunctionExperiment importable with full experiment interface."""

    def test_importable(self):
        from hyperactive.experiment.func import FunctionExperiment

        assert FunctionExperiment is not None

    def test_is_base_experiment_subclass(self):
        from hyperactive.base import BaseExperiment
        from hyperactive.experiment.func import FunctionExperiment

        assert issubclass(FunctionExperiment, BaseExperiment)

    @pytest.mark.parametrize("method", EXPERIMENT_METHODS)
    def test_has_experiment_method(self, method):
        from hyperactive.experiment.func import FunctionExperiment

        assert hasattr(FunctionExperiment, method)


class TestBenchmarkImports:
    """All benchmark experiments importable from hyperactive.experiment.bench."""

    @pytest.mark.parametrize("name", BENCHMARK_NAMES)
    def test_importable(self, name):
        import hyperactive.experiment.bench as mod

        assert hasattr(mod, name), f"{name} not in experiment.bench"

    @pytest.mark.parametrize("name", BENCHMARK_NAMES)
    def test_is_base_experiment_subclass(self, name):
        import hyperactive.experiment.bench as mod
        from hyperactive.base import BaseExperiment

        cls = getattr(mod, name)
        assert issubclass(cls, BaseExperiment)

    @pytest.mark.parametrize("name", BENCHMARK_NAMES)
    def test_has_experiment_methods(self, name):
        import hyperactive.experiment.bench as mod

        cls = getattr(mod, name)
        for method in EXPERIMENT_METHODS:
            assert hasattr(cls, method), f"{name} missing {method}"


class TestExperimentIntegrationImports:
    """All ML experiment wrappers from hyperactive.experiment.integrations."""

    @pytest.mark.parametrize("name", EXPERIMENT_INTEGRATION_NAMES)
    def test_importable(self, name):
        import hyperactive.experiment.integrations as mod

        assert hasattr(mod, name), f"{name} not in experiment.integrations"

    @pytest.mark.parametrize("name", EXPERIMENT_INTEGRATION_NAMES)
    def test_is_base_experiment_subclass(self, name):
        import hyperactive.experiment.integrations as mod
        from hyperactive.base import BaseExperiment

        cls = getattr(mod, name)
        assert issubclass(cls, BaseExperiment)

    @pytest.mark.parametrize("name", EXPERIMENT_INTEGRATION_NAMES)
    def test_has_experiment_methods(self, name):
        import hyperactive.experiment.integrations as mod

        cls = getattr(mod, name)
        for method in EXPERIMENT_METHODS:
            assert hasattr(cls, method), f"{name} missing {method}"


class TestIntegrationWrapperImports:
    """Integration wrappers importable from documented paths."""

    def test_opt_cv_from_integrations(self):
        from hyperactive.integrations import OptCV

        assert OptCV is not None

    def test_opt_cv_from_sklearn(self):
        from hyperactive.integrations.sklearn import OptCV

        assert OptCV is not None

    def test_opt_cv_same_class_both_paths(self):
        from hyperactive.integrations import OptCV as A
        from hyperactive.integrations.sklearn import OptCV as B

        assert A is B

    def test_opt_cv_has_sklearn_api(self):
        from hyperactive.integrations.sklearn import OptCV

        for method in ["fit", "predict", "score", "get_params", "set_params"]:
            assert hasattr(OptCV, method), f"OptCV missing {method}"

    def test_forecasting_opt_cv(self):
        from hyperactive.integrations.sktime import ForecastingOptCV

        assert ForecastingOptCV is not None

    def test_tsc_opt_cv(self):
        from hyperactive.integrations.sktime import TSCOptCV

        assert TSCOptCV is not None

    def test_proba_reg_opt_cv(self):
        from hyperactive.integrations.skpro import ProbaRegOptCV

        assert ProbaRegOptCV is not None


class TestRegistryAndUtils:
    """Registry discovery function and utility importable and callable."""

    def test_all_objects_importable(self):
        from hyperactive._registry import all_objects

        assert callable(all_objects)

    def test_check_estimator_importable(self):
        from hyperactive.utils import check_estimator

        assert callable(check_estimator)


class TestRegistryCompleteness:
    """Registry contains exactly the optimizer and experiment classes we expect.

    Fails if a class is added to the package without updating the name lists
    above, or if an expected class disappears from the registry.
    """

    EXPECTED_OPTIMIZERS = sorted(
        GFO_OPTIMIZER_NAMES + OPTUNA_OPTIMIZER_NAMES + SK_OPTIMIZER_NAMES
    )

    EXPECTED_EXPERIMENTS = sorted(
        ["FunctionExperiment"] + BENCHMARK_NAMES + EXPERIMENT_INTEGRATION_NAMES
    )

    def test_all_expected_optimizers_registered(self):
        from hyperactive._registry import all_objects

        registered = all_objects(object_types="optimizer", return_names=True)
        registered_names = {name for name, _ in registered}

        missing = set(self.EXPECTED_OPTIMIZERS) - registered_names
        assert not missing, f"Expected optimizers missing from registry: {missing}"

    def test_no_untracked_optimizers(self):
        from hyperactive._registry import all_objects

        registered = all_objects(object_types="optimizer", return_names=True)
        registered_names = {name for name, _ in registered}

        unexpected = registered_names - set(self.EXPECTED_OPTIMIZERS)
        assert not unexpected, (
            f"Registry has optimizers not in smoke test spec: {unexpected}. "
            f"Update the name lists in this file."
        )

    def test_all_expected_experiments_registered(self):
        from hyperactive._registry import all_objects

        registered = all_objects(object_types="experiment", return_names=True)
        registered_names = {name for name, _ in registered}

        missing = set(self.EXPECTED_EXPERIMENTS) - registered_names
        assert not missing, f"Expected experiments missing from registry: {missing}"

    def test_no_untracked_experiments(self):
        from hyperactive._registry import all_objects

        registered = all_objects(object_types="experiment", return_names=True)
        registered_names = {name for name, _ in registered}

        unexpected = registered_names - set(self.EXPECTED_EXPERIMENTS)
        assert not unexpected, (
            f"Registry has experiments not in smoke test spec: {unexpected}. "
            f"Update the name lists in this file."
        )
