"""AST-based feature engineering for algorithm selection.

This subpackage provides tools for extracting features from objective
functions and search spaces using Abstract Syntax Tree (AST) analysis.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from .ast_features import (
    ASTFeatureExtractor,
    ASTFeatures,
    extract_ast_features,
)
from .search_space_features import (
    SearchSpaceFeatureExtractor,
    SearchSpaceFeatures,
    extract_search_space_features,
)
from .source_expander import (
    EXTERNAL_LIBRARIES,
    FunctionResolver,
    SourceExpander,
    get_expanded_source,
)

__all__ = [
    # Source expansion
    "SourceExpander",
    "FunctionResolver",
    "get_expanded_source",
    "EXTERNAL_LIBRARIES",
    # AST features
    "ASTFeatureExtractor",
    "ASTFeatures",
    "extract_ast_features",
    # Search space features
    "SearchSpaceFeatureExtractor",
    "SearchSpaceFeatures",
    "extract_search_space_features",
]
