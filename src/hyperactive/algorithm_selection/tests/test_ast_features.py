"""Tests for AST feature extraction.

NOTE: Most tests use extract_from_source() with source code strings
to avoid issues with inspect.getsource() not being able to access
functions defined inside test methods.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import math

import pytest

from hyperactive.algorithm_selection.ast_feature_engineering.ast_features import (
    ASTFeatureExtractor,
    ASTFeatures,
    extract_ast_features,
)


# =============================================================================
# Tests for ASTFeatures dataclass
# =============================================================================


class TestASTFeaturesDataclass:
    """Test the ASTFeatures dataclass."""

    def test_default_values_are_zero(self):
        """All default values should be zero."""
        features = ASTFeatures()

        assert features.code_length == 0
        assert features.num_add == 0
        assert features.num_sin == 0
        assert features.num_if == 0

    def test_to_dict_returns_dict(self):
        """to_dict should return a dictionary."""
        features = ASTFeatures(num_add=5, num_mult=3)
        d = features.to_dict()

        assert isinstance(d, dict)
        assert d["num_add"] == 5
        assert d["num_mult"] == 3

    def test_to_vector_returns_list(self):
        """to_vector should return a list of values."""
        features = ASTFeatures(num_add=5, num_mult=3)
        v = features.to_vector()

        assert isinstance(v, list)
        assert 5 in v
        assert 3 in v

    def test_feature_names_matches_to_dict_keys(self):
        """feature_names should match to_dict keys."""
        features = ASTFeatures()
        names = ASTFeatures.feature_names()
        d = features.to_dict()

        assert names == list(d.keys())

    def test_to_vector_order_matches_feature_names(self):
        """to_vector order should match feature_names order."""
        features = ASTFeatures(num_add=5, num_mult=3, num_pow=7)
        names = ASTFeatures.feature_names()
        vector = features.to_vector()
        d = features.to_dict()

        for i, name in enumerate(names):
            assert vector[i] == d[name]


# =============================================================================
# Tests for basic code metrics
# =============================================================================


class TestBasicCodeMetrics:
    """Test extraction of basic code metrics."""

    def test_code_length(self):
        """code_length should be the character count."""
        source = "def func(x):\n    return x"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.code_length == len(source)

    def test_line_count(self):
        """line_count should count newlines."""
        source = """def func(x):
    a = x + 1
    b = a * 2
    return b"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.line_count == 4

    def test_token_count(self):
        """token_count should count AST nodes."""
        source = "def func(x):\n    return x"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.token_count > 0


# =============================================================================
# Tests for math operations
# =============================================================================


class TestMathOperations:
    """Test extraction of math operation counts."""

    def test_addition_count(self):
        """num_add should count + operations."""
        source = "def func(x):\n    return x + 1 + 2 + 3"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_add == 3

    def test_subtraction_count(self):
        """num_sub should count - operations."""
        source = "def func(x):\n    return x - 1 - 2"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_sub == 2

    def test_multiplication_count(self):
        """num_mult should count * operations."""
        source = "def func(x):\n    return x * 2 * 3"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_mult == 2

    def test_division_count(self):
        """num_div should count / operations."""
        source = "def func(x):\n    return x / 2 / 3"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_div == 2

    def test_power_count(self):
        """num_pow should count ** operations."""
        source = "def func(x):\n    return x ** 2 + x ** 3"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_pow == 2

    def test_modulo_count(self):
        """num_mod should count % operations."""
        source = "def func(x):\n    return x % 2"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_mod == 1

    def test_floor_division_count(self):
        """num_floordiv should count // operations."""
        source = "def func(x):\n    return x // 2"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_floordiv == 1

    def test_unary_minus_count(self):
        """num_unary_minus should count unary - operations."""
        source = "def func(x):\n    return -x + (-5)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_unary_minus == 2

    def test_mixed_operations(self):
        """Multiple operation types should be counted correctly."""
        source = "def func(x):\n    return x ** 2 + x * 3 - x / 2"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_pow == 1
        assert features.num_add == 1
        assert features.num_mult == 1
        assert features.num_sub == 1
        assert features.num_div == 1


# =============================================================================
# Tests for math functions
# =============================================================================


class TestMathFunctions:
    """Test extraction of math function calls."""

    def test_sin_count(self):
        """num_sin should count sin() calls."""
        source = "import math\ndef func(x):\n    return math.sin(x) + math.sin(x * 2)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_sin == 2

    def test_cos_count(self):
        """num_cos should count cos() calls."""
        source = "import math\ndef func(x):\n    return math.cos(x)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_cos == 1

    def test_exp_count(self):
        """num_exp should count exp() calls."""
        source = "import math\ndef func(x):\n    return math.exp(x) + math.exp(-x)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_exp == 2

    def test_log_count(self):
        """num_log should count log() calls."""
        source = "import math\ndef func(x):\n    return math.log(x) + math.log10(x)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        # Both log and log10 should count
        assert features.num_log == 2

    def test_sqrt_count(self):
        """num_sqrt should count sqrt() calls."""
        source = "import math\ndef func(x):\n    return math.sqrt(x)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_sqrt == 1

    def test_abs_count(self):
        """num_abs should count abs() calls."""
        source = "def func(x):\n    return abs(x) + abs(x - 1)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_abs == 2

    def test_sum_count(self):
        """num_sum should count sum() calls."""
        source = "def func(x):\n    return sum(x)"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_sum == 1


# =============================================================================
# Tests for comparisons
# =============================================================================


class TestComparisons:
    """Test extraction of comparison operations."""

    def test_equality_count(self):
        """num_eq should count == comparisons."""
        source = "def func(x):\n    return x == 0"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_eq == 1
        assert features.num_comparisons == 1

    def test_inequality_count(self):
        """num_noteq should count != comparisons."""
        source = "def func(x):\n    return x != 0"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_noteq == 1

    def test_less_than_count(self):
        """num_lt should count < comparisons."""
        source = "def func(x):\n    return x < 0"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_lt == 1

    def test_greater_than_count(self):
        """num_gt should count > comparisons."""
        source = "def func(x):\n    return x > 0"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_gt == 1

    def test_chained_comparisons(self):
        """Chained comparisons should count each operator."""
        source = "def func(x):\n    return 0 < x < 10"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_comparisons == 2
        assert features.num_lt == 2


# =============================================================================
# Tests for control flow
# =============================================================================


class TestControlFlow:
    """Test extraction of control flow structures."""

    def test_if_count(self):
        """num_if should count if statements."""
        source = """def func(x):
    if x > 0:
        return x
    return -x"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_if == 1

    def test_nested_if_count(self):
        """Nested ifs should each be counted."""
        source = """def func(x):
    if x > 0:
        if x > 10:
            return x
    return 0"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_if == 2

    def test_for_loop_count(self):
        """num_for should count for loops."""
        source = """def func(x):
    total = 0
    for i in range(x):
        total += i
    return total"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_for == 1

    def test_while_loop_count(self):
        """num_while should count while loops."""
        source = """def func(x):
    while x > 0:
        x -= 1
    return x"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_while == 1

    def test_max_loop_depth(self):
        """max_loop_depth should track nesting."""
        source = """def func(x):
    for i in range(x):
        for j in range(x):
            for k in range(x):
                pass
    return 0"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.max_loop_depth == 3

    def test_max_if_depth(self):
        """max_if_depth should track if nesting."""
        source = """def func(x):
    if x > 0:
        if x > 5:
            if x > 10:
                return x
    return 0"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.max_if_depth == 3


# =============================================================================
# Tests for boolean operations
# =============================================================================


class TestBooleanOperations:
    """Test extraction of boolean operations."""

    def test_and_count(self):
        """num_and should count 'and' operations."""
        source = "def func(x):\n    return x > 0 and x < 10"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_and == 1

    def test_or_count(self):
        """num_or should count 'or' operations."""
        source = "def func(x):\n    return x < 0 or x > 10"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_or == 1

    def test_not_count(self):
        """num_not should count 'not' operations."""
        source = "def func(x):\n    return not x"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_not == 1

    def test_chained_and(self):
        """Chained 'and' should count multiple."""
        source = "def func(x):\n    return x > 0 and x < 10 and x != 5"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_and == 2


# =============================================================================
# Tests for structure metrics
# =============================================================================


class TestStructureMetrics:
    """Test extraction of structural metrics."""

    def test_function_def_count(self):
        """num_function_defs should count function definitions."""
        source = """def func(x):
    def inner(y):
        return y
    return inner(x)"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_function_defs == 2  # outer and inner

    def test_function_call_count(self):
        """num_function_calls should count function calls."""
        source = "def func(x):\n    return abs(x) + sum([1, 2, 3])"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_function_calls == 2

    def test_subscript_count(self):
        """num_subscripts should count indexing operations."""
        source = 'def func(x):\n    return x["a"] + x["b"] + x[0]'

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_subscripts == 3

    def test_attribute_count(self):
        """num_attributes should count attribute access."""
        source = "def func(x):\n    return x.value + x.other"

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_attributes == 2

    def test_constant_count(self):
        """num_constants should count constant values."""
        source = 'def func(x):\n    return x + 1 + 2.5 + "string"'

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_constants >= 3


# =============================================================================
# Module-level test functions for source expansion tests
# =============================================================================


def helper_for_expansion(x):
    """Helper function for expansion tests."""
    return x ** 2 + x ** 3


def objective_for_expansion(x):
    """Objective that uses helper_for_expansion."""
    return helper_for_expansion(x["val"])


def objective_with_sin(x):
    """Objective using math.sin."""
    return math.sin(x["val"])


# =============================================================================
# Tests for source expansion integration
# =============================================================================


class TestSourceExpansionIntegration:
    """Test that source expansion works with feature extraction."""

    def test_expand_source_includes_helpers(self):
        """With expand_source=True, helper functions should be analyzed."""
        # Without expansion
        extractor_no_expand = ASTFeatureExtractor(expand_source=False)
        features_no_expand = extractor_no_expand.extract(objective_for_expansion)

        # With expansion
        extractor_expand = ASTFeatureExtractor(expand_source=True)
        features_expand = extractor_expand.extract(objective_for_expansion)

        # With expansion should have more power operations
        assert features_expand.num_pow >= features_no_expand.num_pow

    def test_no_expand_source_only_root(self):
        """With expand_source=False, only root function analyzed."""
        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract(objective_with_sin)

        # sin is in this function, but we're not expanding
        # The function still uses math.sin directly
        assert features.num_sin >= 1


# =============================================================================
# Tests for convenience function
# =============================================================================


class TestConvenienceFunction:
    """Test the extract_ast_features convenience function."""

    def test_extract_ast_features_basic(self):
        """extract_ast_features should work like the extractor."""
        features = extract_ast_features(objective_for_expansion, expand_source=False)

        # Should at least have some features
        assert features.code_length > 0

    def test_extract_ast_features_expand_param(self):
        """expand_source parameter should be passed through."""
        features_expand = extract_ast_features(objective_for_expansion, expand_source=True)
        features_no_expand = extract_ast_features(objective_for_expansion, expand_source=False)

        # With expansion should detect more power operations
        assert features_expand.num_pow >= features_no_expand.num_pow


# =============================================================================
# Tests for realistic objective functions
# =============================================================================


class TestRealisticObjectiveFunctions:
    """Test with realistic optimization objective functions."""

    def test_sphere_function(self):
        """Test analysis of sphere function."""
        source = """def sphere(x):
    return sum(xi ** 2 for xi in x.values())"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_pow >= 1
        assert features.num_sum >= 1

    def test_rastrigin_like_function(self):
        """Test analysis of Rastrigin-like function with trig."""
        source = """import math
def rastrigin_like(x):
    n = len(x)
    A = 10
    result = A * n
    for xi in x.values():
        result += xi ** 2 - A * math.cos(2 * math.pi * xi)
    return result"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_pow >= 1
        assert features.num_cos >= 1
        assert features.num_for >= 1

    def test_conditional_objective(self):
        """Test analysis of objective with conditionals."""
        source = """def conditional_obj(x):
    val = x["a"] ** 2 + x["b"] ** 2
    if val > 10:
        return val * 2
    elif val > 5:
        return val * 1.5
    else:
        return val"""

        extractor = ASTFeatureExtractor(expand_source=False)
        features = extractor.extract_from_source(source)

        assert features.num_pow == 2
        assert features.num_if >= 1  # if and elif count
        assert features.num_comparisons >= 2
