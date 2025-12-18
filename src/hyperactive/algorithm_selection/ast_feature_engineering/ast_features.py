"""AST-based feature extraction from Python source code.

This module extracts numerical features from Python source code by
analyzing the Abstract Syntax Tree (AST). These features characterize
the structure and complexity of objective functions for algorithm selection.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .source_expander import get_expanded_source


@dataclass
class ASTFeatures:
    """Container for AST-extracted features.

    Attributes
    ----------
    code_length : int
        Total number of characters in the source code.
    line_count : int
        Number of lines in the source code.
    token_count : int
        Approximate number of tokens (AST nodes).

    Math Operations
    ---------------
    num_add : int
        Count of addition operations (+).
    num_sub : int
        Count of subtraction operations (-).
    num_mult : int
        Count of multiplication operations (*).
    num_div : int
        Count of division operations (/).
    num_pow : int
        Count of power operations (**).
    num_mod : int
        Count of modulo operations (%).
    num_floordiv : int
        Count of floor division operations (//).
    num_unary_minus : int
        Count of unary minus operations.

    Math Functions
    --------------
    num_sin : int
        Count of sin function calls.
    num_cos : int
        Count of cos function calls.
    num_tan : int
        Count of tan function calls.
    num_exp : int
        Count of exp function calls.
    num_log : int
        Count of log function calls.
    num_sqrt : int
        Count of sqrt function calls.
    num_abs : int
        Count of abs function calls.
    num_sum : int
        Count of sum function calls.
    num_mean : int
        Count of mean function calls.

    Comparisons
    -----------
    num_comparisons : int
        Total count of comparison operations.
    num_eq : int
        Count of equality comparisons (==).
    num_noteq : int
        Count of inequality comparisons (!=).
    num_lt : int
        Count of less-than comparisons (<).
    num_gt : int
        Count of greater-than comparisons (>).
    num_lte : int
        Count of less-than-or-equal comparisons (<=).
    num_gte : int
        Count of greater-than-or-equal comparisons (>=).

    Control Flow
    ------------
    num_if : int
        Count of if statements.
    num_for : int
        Count of for loops.
    num_while : int
        Count of while loops.
    num_try : int
        Count of try/except blocks.
    max_loop_depth : int
        Maximum nesting depth of loops.
    max_if_depth : int
        Maximum nesting depth of if statements.

    Structure
    ---------
    num_function_defs : int
        Count of function definitions.
    num_function_calls : int
        Count of function calls.
    num_subscripts : int
        Count of subscript operations (indexing).
    num_attributes : int
        Count of attribute accesses.
    num_names : int
        Count of name references.
    num_constants : int
        Count of constant values.
    max_expression_depth : int
        Maximum depth of expression nesting.

    Boolean Operations
    ------------------
    num_and : int
        Count of logical and operations.
    num_or : int
        Count of logical or operations.
    num_not : int
        Count of logical not operations.
    """

    # Code metrics
    code_length: int = 0
    line_count: int = 0
    token_count: int = 0

    # Math operations
    num_add: int = 0
    num_sub: int = 0
    num_mult: int = 0
    num_div: int = 0
    num_pow: int = 0
    num_mod: int = 0
    num_floordiv: int = 0
    num_unary_minus: int = 0

    # Math functions
    num_sin: int = 0
    num_cos: int = 0
    num_tan: int = 0
    num_exp: int = 0
    num_log: int = 0
    num_sqrt: int = 0
    num_abs: int = 0
    num_sum: int = 0
    num_mean: int = 0

    # Comparisons
    num_comparisons: int = 0
    num_eq: int = 0
    num_noteq: int = 0
    num_lt: int = 0
    num_gt: int = 0
    num_lte: int = 0
    num_gte: int = 0

    # Control flow
    num_if: int = 0
    num_for: int = 0
    num_while: int = 0
    num_try: int = 0
    max_loop_depth: int = 0
    max_if_depth: int = 0

    # Structure
    num_function_defs: int = 0
    num_function_calls: int = 0
    num_subscripts: int = 0
    num_attributes: int = 0
    num_names: int = 0
    num_constants: int = 0
    max_expression_depth: int = 0

    # Boolean operations
    num_and: int = 0
    num_or: int = 0
    num_not: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert features to dictionary.

        Returns
        -------
        dict
            Dictionary with feature names as keys and counts as values.
        """
        return {
            # Code metrics
            "code_length": self.code_length,
            "line_count": self.line_count,
            "token_count": self.token_count,
            # Math operations
            "num_add": self.num_add,
            "num_sub": self.num_sub,
            "num_mult": self.num_mult,
            "num_div": self.num_div,
            "num_pow": self.num_pow,
            "num_mod": self.num_mod,
            "num_floordiv": self.num_floordiv,
            "num_unary_minus": self.num_unary_minus,
            # Math functions
            "num_sin": self.num_sin,
            "num_cos": self.num_cos,
            "num_tan": self.num_tan,
            "num_exp": self.num_exp,
            "num_log": self.num_log,
            "num_sqrt": self.num_sqrt,
            "num_abs": self.num_abs,
            "num_sum": self.num_sum,
            "num_mean": self.num_mean,
            # Comparisons
            "num_comparisons": self.num_comparisons,
            "num_eq": self.num_eq,
            "num_noteq": self.num_noteq,
            "num_lt": self.num_lt,
            "num_gt": self.num_gt,
            "num_lte": self.num_lte,
            "num_gte": self.num_gte,
            # Control flow
            "num_if": self.num_if,
            "num_for": self.num_for,
            "num_while": self.num_while,
            "num_try": self.num_try,
            "max_loop_depth": self.max_loop_depth,
            "max_if_depth": self.max_if_depth,
            # Structure
            "num_function_defs": self.num_function_defs,
            "num_function_calls": self.num_function_calls,
            "num_subscripts": self.num_subscripts,
            "num_attributes": self.num_attributes,
            "num_names": self.num_names,
            "num_constants": self.num_constants,
            "max_expression_depth": self.max_expression_depth,
            # Boolean operations
            "num_and": self.num_and,
            "num_or": self.num_or,
            "num_not": self.num_not,
        }

    def to_vector(self) -> list[int]:
        """Convert features to a list (for ML models).

        Returns
        -------
        list of int
            Feature values in consistent order.
        """
        return list(self.to_dict().values())

    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names in consistent order.

        Returns
        -------
        list of str
            Feature names matching to_vector() order.
        """
        return list(ASTFeatures().to_dict().keys())


class ASTFeatureExtractor:
    """Extract features from Python source code via AST analysis.

    This class walks the AST of source code and counts various
    structural elements that may be predictive of optimization
    landscape characteristics.

    Parameters
    ----------
    expand_source : bool, default=True
        If True, expand function calls to include called functions.
    max_expansion_depth : int, default=10
        Maximum depth for source expansion.

    Examples
    --------
    >>> def objective(x):
    ...     return x["a"] ** 2 + x["b"] ** 2
    >>> extractor = ASTFeatureExtractor(expand_source=False)
    >>> features = extractor.extract(objective)
    >>> features.num_pow
    2
    >>> features.num_add
    1
    """

    # Mapping from AST operator types to feature names
    BINOP_MAP = {
        ast.Add: "num_add",
        ast.Sub: "num_sub",
        ast.Mult: "num_mult",
        ast.Div: "num_div",
        ast.Pow: "num_pow",
        ast.Mod: "num_mod",
        ast.FloorDiv: "num_floordiv",
    }

    CMPOP_MAP = {
        ast.Eq: "num_eq",
        ast.NotEq: "num_noteq",
        ast.Lt: "num_lt",
        ast.Gt: "num_gt",
        ast.LtE: "num_lte",
        ast.GtE: "num_gte",
    }

    BOOLOP_MAP = {
        ast.And: "num_and",
        ast.Or: "num_or",
    }

    # Math function names to track
    MATH_FUNCTIONS = {
        "sin": "num_sin",
        "cos": "num_cos",
        "tan": "num_tan",
        "exp": "num_exp",
        "log": "num_log",
        "log10": "num_log",
        "log2": "num_log",
        "sqrt": "num_sqrt",
        "abs": "num_abs",
        "sum": "num_sum",
        "mean": "num_mean",
        "average": "num_mean",
    }

    def __init__(self, expand_source: bool = True, max_expansion_depth: int = 10):
        self.expand_source = expand_source
        self.max_expansion_depth = max_expansion_depth

    def extract(self, func: Callable) -> ASTFeatures:
        """Extract features from a function.

        Parameters
        ----------
        func : Callable
            The function to analyze.

        Returns
        -------
        ASTFeatures
            Extracted features.
        """
        if self.expand_source:
            source = get_expanded_source(func, max_depth=self.max_expansion_depth)
        else:
            import inspect

            try:
                source = inspect.getsource(func)
            except (TypeError, OSError):
                source = ""

        return self.extract_from_source(source)

    def extract_from_source(self, source: str) -> ASTFeatures:
        """Extract features from source code string.

        Parameters
        ----------
        source : str
            Python source code.

        Returns
        -------
        ASTFeatures
            Extracted features.
        """
        features = ASTFeatures()

        if not source:
            return features

        # Basic code metrics
        features.code_length = len(source)
        features.line_count = source.count("\n") + 1

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return features

        # Count tokens (AST nodes)
        features.token_count = sum(1 for _ in ast.walk(tree))

        # Walk the AST and extract features
        self._extract_from_tree(tree, features)

        # Calculate depth metrics
        features.max_expression_depth = self._max_expression_depth(tree)
        features.max_loop_depth = self._max_nesting_depth(tree, (ast.For, ast.While))
        features.max_if_depth = self._max_nesting_depth(tree, (ast.If,))

        return features

    def _extract_from_tree(self, tree: ast.AST, features: ASTFeatures):
        """Walk AST and count features.

        Parameters
        ----------
        tree : ast.AST
            The AST to analyze.
        features : ASTFeatures
            Features object to update.
        """
        for node in ast.walk(tree):
            # Binary operations
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in self.BINOP_MAP:
                    attr = self.BINOP_MAP[op_type]
                    setattr(features, attr, getattr(features, attr) + 1)

            # Unary operations
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    features.num_unary_minus += 1
                elif isinstance(node.op, ast.Not):
                    features.num_not += 1

            # Comparisons
            elif isinstance(node, ast.Compare):
                features.num_comparisons += len(node.ops)
                for op in node.ops:
                    op_type = type(op)
                    if op_type in self.CMPOP_MAP:
                        attr = self.CMPOP_MAP[op_type]
                        setattr(features, attr, getattr(features, attr) + 1)

            # Boolean operations
            elif isinstance(node, ast.BoolOp):
                op_type = type(node.op)
                if op_type in self.BOOLOP_MAP:
                    attr = self.BOOLOP_MAP[op_type]
                    # Count is number of values - 1 (a and b and c = 2 ands)
                    count = len(node.values) - 1
                    setattr(features, attr, getattr(features, attr) + count)

            # Control flow
            elif isinstance(node, ast.If):
                features.num_if += 1
            elif isinstance(node, ast.For):
                features.num_for += 1
            elif isinstance(node, ast.While):
                features.num_while += 1
            elif isinstance(node, ast.Try):
                features.num_try += 1

            # Function definitions
            elif isinstance(node, ast.FunctionDef):
                features.num_function_defs += 1

            # Function calls
            elif isinstance(node, ast.Call):
                features.num_function_calls += 1
                self._check_math_function(node, features)

            # Subscripts (indexing)
            elif isinstance(node, ast.Subscript):
                features.num_subscripts += 1

            # Attribute access
            elif isinstance(node, ast.Attribute):
                features.num_attributes += 1

            # Names
            elif isinstance(node, ast.Name):
                features.num_names += 1

            # Constants
            elif isinstance(node, ast.Constant):
                features.num_constants += 1

    def _check_math_function(self, node: ast.Call, features: ASTFeatures):
        """Check if a call is a known math function.

        Parameters
        ----------
        node : ast.Call
            The call node.
        features : ASTFeatures
            Features to update.
        """
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name and func_name in self.MATH_FUNCTIONS:
            attr = self.MATH_FUNCTIONS[func_name]
            setattr(features, attr, getattr(features, attr) + 1)

    def _max_expression_depth(self, tree: ast.AST) -> int:
        """Calculate maximum expression nesting depth.

        Parameters
        ----------
        tree : ast.AST
            The AST to analyze.

        Returns
        -------
        int
            Maximum expression depth.
        """
        max_depth = 0

        def walk_depth(node: ast.AST, depth: int):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    walk_depth(child, depth + 1)
                else:
                    walk_depth(child, depth)

        walk_depth(tree, 0)
        return max_depth

    def _max_nesting_depth(
        self, tree: ast.AST, node_types: tuple[type, ...]
    ) -> int:
        """Calculate maximum nesting depth for specific node types.

        Parameters
        ----------
        tree : ast.AST
            The AST to analyze.
        node_types : tuple of type
            AST node types to track (e.g., ast.For, ast.While).

        Returns
        -------
        int
            Maximum nesting depth.
        """
        max_depth = 0

        def walk_depth(node: ast.AST, depth: int):
            nonlocal max_depth

            current_depth = depth
            if isinstance(node, node_types):
                current_depth = depth + 1
                max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(node):
                walk_depth(child, current_depth)

        walk_depth(tree, 0)
        return max_depth


def extract_ast_features(func: Callable, expand_source: bool = True) -> ASTFeatures:
    """Convenience function to extract AST features from a function.

    Parameters
    ----------
    func : Callable
        The function to analyze.
    expand_source : bool, default=True
        If True, expand function calls to include called functions.

    Returns
    -------
    ASTFeatures
        Extracted features.

    Examples
    --------
    >>> def objective(x):  # doctest: +SKIP
    ...     return x["a"] ** 2 + x["b"] ** 2
    >>> features = extract_ast_features(objective)  # doctest: +SKIP
    >>> features.num_pow  # doctest: +SKIP
    2
    """
    extractor = ASTFeatureExtractor(expand_source=expand_source)
    return extractor.extract(func)
