"""Comprehensive tests for source_expander module.

This module thoroughly tests the source expansion and import following
functionality. The behavior must be well-defined and predictable.

NOTE: Functions used for testing source expansion MUST be defined at
module level because inspect.getsource() cannot access source code
of functions defined inside other functions (like test methods).
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import math

import pytest

from hyperactive.algorithm_selection.ast_feature_engineering.source_expander import (
    EXTERNAL_LIBRARIES,
    FunctionResolver,
    SourceExpander,
    get_expanded_source,
)


# =============================================================================
# Module-level test functions (required for source inspection)
# =============================================================================


def simple_helper(x):
    """A simple helper function for testing."""
    return x * 2


def nested_helper_outer(x):
    """Outer helper that calls inner helper."""
    return nested_helper_inner(x) + 1


def nested_helper_inner(x):
    """Inner helper function."""
    return x ** 2


def recursive_func(n):
    """A recursive function for cycle detection testing."""
    if n <= 0:
        return 0
    return n + recursive_func(n - 1)


def mutually_recursive_a(n):
    """Mutually recursive function A."""
    if n <= 0:
        return 0
    return mutually_recursive_b(n - 1)


def mutually_recursive_b(n):
    """Mutually recursive function B."""
    if n <= 0:
        return 1
    return mutually_recursive_a(n - 1)


# Test objective functions - defined at module level for source access
def objective_simple(x):
    """Simple objective with no external calls."""
    return x["a"] ** 2


def objective_empty(x):
    """Empty objective function."""
    pass


def objective_with_simple_helper(x):
    """Objective that calls simple_helper."""
    return simple_helper(x["val"])


def objective_with_chained_helpers(x):
    """Objective that calls nested_helper_outer (which calls nested_helper_inner)."""
    return nested_helper_outer(x["val"])


def objective_with_math(x):
    """Objective using math module (external)."""
    return math.sqrt(x["val"])


def objective_with_local_import(x):
    """Objective with local import."""
    from math import sqrt
    return sqrt(x["val"])


def objective_with_local_import_module(x):
    """Objective with local module import."""
    import math
    return math.sqrt(x["val"])


def objective_with_local_import_alias(x):
    """Objective with local import with alias."""
    from math import sqrt as sq
    return sq(x["val"])


def objective_with_nested_def(x):
    """Objective with nested function definition."""
    def inner_helper(y):
        return y ** 2
    return inner_helper(x["val"])


def objective_with_deeply_nested(x):
    """Objective with deeply nested function definitions."""
    def level1(a):
        def level2(b):
            return b * 2
        return level2(a) + 1
    return level1(x["val"])


def objective_with_builtins(x):
    """Objective using built-in functions."""
    return sum(x.values()) + len(x)


def objective_with_recursive(x):
    """Objective calling recursive function."""
    return recursive_func(x["val"])


def objective_with_mutual_recursion(x):
    """Objective calling mutually recursive function."""
    return mutually_recursive_a(x["val"])


def preprocess_data(data):
    """Preprocess input data - used in complex objective."""
    return {k: v * 2 for k, v in data.items()}


def compute_penalty(x):
    """Compute a penalty term - used in complex objective."""
    return sum(v ** 2 for v in x.values() if v < 0)


def objective_complex(params):
    """Complex objective with multiple helpers."""
    processed = preprocess_data(params)
    base_score = sum(processed.values())
    penalty = compute_penalty(processed)
    return base_score - penalty


def objective_with_mixed_imports(x):
    """Objective with both local and global imports."""
    from collections import Counter
    counts = Counter(x.values())
    return math.sqrt(sum(counts.values()))


# =============================================================================
# Tests for SourceExpander - Basic functionality
# =============================================================================


class TestSourceExpanderBasic:
    """Test basic source expansion functionality."""

    def test_expand_simple_function(self):
        """Test expanding a function with no external calls."""
        expander = SourceExpander()
        source = expander.expand(objective_simple)

        assert "def objective_simple" in source
        assert "x[" in source or 'x["a"]' in source
        assert "** 2" in source

    def test_expand_returns_string(self):
        """Expanded source should always be a string."""
        expander = SourceExpander()
        source = expander.expand(objective_simple)

        assert isinstance(source, str)

    def test_expand_empty_function(self):
        """Test expanding a function that does nothing."""
        expander = SourceExpander()
        source = expander.expand(objective_empty)

        assert "def objective_empty" in source
        assert "pass" in source

    def test_resolution_log_populated(self):
        """Resolution log should be populated after expand."""
        expander = SourceExpander()
        expander.expand(objective_simple)

        assert len(expander.resolution_log) > 0
        assert expander.resolution_log[0]["function"] == "objective_simple"
        assert expander.resolution_log[0]["success"] is True


# =============================================================================
# Tests for __globals__ resolution
# =============================================================================


class TestGlobalsResolution:
    """Test resolution of functions from __globals__."""

    def test_resolve_same_module_function(self):
        """Functions defined in the same module should be expanded."""
        expander = SourceExpander()
        source = expander.expand(objective_with_simple_helper)

        # Both functions should be in the output
        assert "def objective_with_simple_helper" in source
        assert "def simple_helper" in source
        assert "x * 2" in source

    def test_resolve_chained_calls(self):
        """Chained function calls should all be expanded."""
        expander = SourceExpander()
        source = expander.expand(objective_with_chained_helpers)

        # All three functions should be present
        assert "def objective_with_chained_helpers" in source
        assert "def nested_helper_outer" in source
        assert "def nested_helper_inner" in source

    def test_globals_with_imported_function(self):
        """Module-level imports should be in __globals__."""
        expander = SourceExpander()
        source = expander.expand(objective_with_math)

        # math.sqrt is external, should not be expanded
        assert "def objective_with_math" in source
        # math module functions should NOT be expanded
        assert "def sqrt" not in source


# =============================================================================
# Tests for local import resolution
# =============================================================================


class TestLocalImportResolution:
    """Test resolution of imports inside function bodies."""

    def test_local_import_from_statement(self):
        """Local 'from x import y' should be detected."""
        expander = SourceExpander()
        source = expander.expand(objective_with_local_import)

        # sqrt is from math (external), should not be expanded
        assert "def objective_with_local_import" in source
        # Local import syntax should be in the source
        assert "from math import sqrt" in source

    def test_local_import_statement(self):
        """Local 'import x' should be detected."""
        expander = SourceExpander()
        source = expander.expand(objective_with_local_import_module)

        assert "def objective_with_local_import_module" in source
        assert "import math" in source

    def test_local_import_with_alias(self):
        """Local imports with aliases should work."""
        expander = SourceExpander()
        source = expander.expand(objective_with_local_import_alias)

        assert "def objective_with_local_import_alias" in source
        assert "as sq" in source


# =============================================================================
# Tests for nested function handling
# =============================================================================


class TestNestedFunctions:
    """Test handling of nested function definitions."""

    def test_nested_function_in_source(self):
        """Nested functions should be included in expanded source."""
        expander = SourceExpander()
        source = expander.expand(objective_with_nested_def)

        # Both outer and nested function should be in source
        assert "def objective_with_nested_def" in source
        assert "def inner_helper" in source
        assert "y ** 2" in source

    def test_deeply_nested_functions(self):
        """Multiple levels of nesting should work."""
        expander = SourceExpander()
        source = expander.expand(objective_with_deeply_nested)

        assert "def objective_with_deeply_nested" in source
        assert "def level1" in source
        assert "def level2" in source


# =============================================================================
# Tests for external library detection
# =============================================================================


class TestExternalLibraryDetection:
    """Test that external libraries are correctly identified and not expanded."""

    def test_numpy_is_external(self):
        """numpy should be recognized as external."""
        assert "numpy" in EXTERNAL_LIBRARIES
        assert "np" in EXTERNAL_LIBRARIES

    def test_sklearn_is_external(self):
        """sklearn should be recognized as external."""
        assert "sklearn" in EXTERNAL_LIBRARIES

    def test_torch_is_external(self):
        """torch should be recognized as external."""
        assert "torch" in EXTERNAL_LIBRARIES

    def test_builtin_functions_not_expanded(self):
        """Built-in functions like len, sum should not be expanded."""
        expander = SourceExpander()
        source = expander.expand(objective_with_builtins)

        # Should only contain objective, not built-in definitions
        assert "def objective_with_builtins" in source
        assert source.count("def ") == 1  # Only one function definition

    def test_builtin_is_external(self):
        """Built-in functions should be detected as external."""
        expander = SourceExpander()

        assert expander._is_external(len)
        assert expander._is_external(sum)
        assert expander._is_external(print)

    def test_c_extension_is_external(self):
        """C extension functions should be detected as external."""
        expander = SourceExpander()

        # math.sqrt is typically a C function
        assert expander._is_external(math.sqrt)


# =============================================================================
# Tests for cycle detection
# =============================================================================


class TestCycleDetection:
    """Test that recursive and mutually recursive functions don't cause infinite loops."""

    def test_recursive_function_no_infinite_loop(self):
        """Recursive functions should be handled without infinite loop."""
        expander = SourceExpander()
        # This should complete without hanging
        source = expander.expand(objective_with_recursive)

        assert "def objective_with_recursive" in source
        assert "def recursive_func" in source
        # Should only appear once despite recursive calls
        assert source.count("def recursive_func") == 1

    def test_mutually_recursive_functions(self):
        """Mutually recursive functions should be handled."""
        expander = SourceExpander()
        source = expander.expand(objective_with_mutual_recursion)

        assert "def objective_with_mutual_recursion" in source
        assert "def mutually_recursive_a" in source
        assert "def mutually_recursive_b" in source
        # Each should appear exactly once
        assert source.count("def mutually_recursive_a") == 1
        assert source.count("def mutually_recursive_b") == 1

    def test_visited_tracking(self):
        """Visited set should track processed functions."""
        expander = SourceExpander()
        expander.expand(objective_with_simple_helper)

        # visited should have entries
        assert len(expander.visited) >= 2  # At least objective and simple_helper


# =============================================================================
# Tests for max depth limiting
# =============================================================================


class TestMaxDepthLimiting:
    """Test that max_depth parameter limits expansion."""

    def test_max_depth_zero_only_root(self):
        """max_depth=0 should only include the root function."""
        expander = SourceExpander(max_depth=0)
        source = expander.expand(objective_with_simple_helper)

        # Only objective should be present
        assert "def objective_with_simple_helper" in source
        # simple_helper should NOT be expanded (depth would be 1)
        assert "def simple_helper" not in source

    def test_max_depth_one_includes_direct_calls(self):
        """max_depth=1 should include direct calls but not their calls."""
        expander = SourceExpander(max_depth=1)
        source = expander.expand(objective_with_chained_helpers)

        # objective and nested_helper_outer should be present
        assert "def objective_with_chained_helpers" in source
        assert "def nested_helper_outer" in source
        # nested_helper_inner should NOT be expanded (depth would be 2)
        assert "def nested_helper_inner" not in source

    def test_default_max_depth_sufficient(self):
        """Default max_depth should handle reasonable nesting."""
        expander = SourceExpander()  # default max_depth=10
        source = expander.expand(objective_with_chained_helpers)

        # All functions should be expanded
        assert "def nested_helper_inner" in source


# =============================================================================
# Tests for FunctionResolver class
# =============================================================================


class TestFunctionResolver:
    """Test the FunctionResolver class directly."""

    def test_resolve_simple_name(self):
        """Simple function names should be resolved from __globals__."""
        import inspect
        import ast

        source = inspect.getsource(objective_with_simple_helper)
        resolver = FunctionResolver(objective_with_simple_helper, source)

        tree = ast.parse(source)
        call_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

        assert len(call_nodes) > 0
        # The call to simple_helper should be resolvable
        resolved = resolver.resolve_call(call_nodes[0])
        assert resolved is simple_helper

    def test_local_imports_property(self):
        """local_imports property should extract imports from function body."""
        import inspect

        source = inspect.getsource(objective_with_local_import)
        resolver = FunctionResolver(objective_with_local_import, source)

        imports = resolver.local_imports
        assert "sqrt" in imports
        assert imports["sqrt"] == ("math", "sqrt")

    def test_nested_functions_property(self):
        """nested_functions property should extract nested function definitions."""
        import inspect

        source = inspect.getsource(objective_with_nested_def)
        resolver = FunctionResolver(objective_with_nested_def, source)

        nested = resolver.nested_functions
        assert "inner_helper" in nested

    def test_resolve_attribute_access(self):
        """Attribute access like module.func should be handled."""
        import inspect
        import ast

        source = inspect.getsource(objective_with_math)
        resolver = FunctionResolver(objective_with_math, source)

        tree = ast.parse(source)
        call_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

        # math.sqrt call
        resolved = resolver.resolve_call(call_nodes[0])
        # Should resolve to math.sqrt
        assert resolved is math.sqrt


# =============================================================================
# Tests for convenience function
# =============================================================================


class TestConvenienceFunction:
    """Test the get_expanded_source convenience function."""

    def test_get_expanded_source_basic(self):
        """get_expanded_source should work like SourceExpander.expand."""
        source = get_expanded_source(objective_with_simple_helper)

        assert "def objective_with_simple_helper" in source
        assert "def simple_helper" in source

    def test_get_expanded_source_with_max_depth(self):
        """max_depth parameter should be passed through."""
        source = get_expanded_source(objective_with_chained_helpers, max_depth=1)

        assert "def objective_with_chained_helpers" in source
        assert "def nested_helper_outer" in source
        assert "def nested_helper_inner" not in source


# =============================================================================
# Tests for edge cases and error handling
# =============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_lambda_returns_string(self):
        """Lambda functions should return a string (possibly empty)."""
        objective = lambda x: x ** 2  # noqa: E731

        expander = SourceExpander()
        source = expander.expand(objective)

        # Should return a string (possibly empty since lambdas may not have source)
        assert isinstance(source, str)

    def test_state_reset_between_calls(self):
        """State should be reset between expand() calls."""
        expander = SourceExpander()

        source1 = expander.expand(objective_simple)
        visited1 = expander.visited.copy()

        source2 = expander.expand(objective_with_simple_helper)
        visited2 = expander.visited.copy()

        # State should be different between calls
        assert visited1 != visited2
        assert len(expander.resolution_log) > 0


# =============================================================================
# Tests for resolution logging
# =============================================================================


class TestResolutionLogging:
    """Test the resolution logging functionality."""

    def test_log_contains_function_name(self):
        """Log entries should contain function names."""
        expander = SourceExpander()
        expander.expand(objective_simple)

        assert any(
            entry["function"] == "objective_simple"
            for entry in expander.resolution_log
        )

    def test_log_contains_status(self):
        """Log entries should contain status information."""
        expander = SourceExpander()
        expander.expand(objective_with_simple_helper)

        for entry in expander.resolution_log:
            assert "status" in entry
            assert "success" in entry

    def test_log_external_library_status(self):
        """External libraries should be logged with appropriate status."""
        expander = SourceExpander()
        expander.expand(objective_with_math)

        # Should have a log entry for the external library
        external_entries = [
            e for e in expander.resolution_log if e["status"] == "external_library"
        ]
        # math.sqrt should be logged as external
        assert len(external_entries) >= 1


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_objective_function(self):
        """Test a realistic complex objective function."""
        expander = SourceExpander()
        source = expander.expand(objective_complex)

        # All user-defined functions should be expanded
        assert "def objective_complex" in source
        assert "def preprocess_data" in source
        assert "def compute_penalty" in source

    def test_mixed_local_and_global_imports(self):
        """Test function with both local and global imports."""
        expander = SourceExpander()
        source = expander.expand(objective_with_mixed_imports)

        # Function should be captured
        assert "def objective_with_mixed_imports" in source
        # Local import should be visible in source
        assert "from collections import Counter" in source
