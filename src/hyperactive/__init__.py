"""Hyperactive - A hyperparameter optimization and meta-learning library.

This package provides tools for hyperparameter optimization using various
optimization algorithms including random search, grid search, Bayesian
optimization, and many others.

Author: Simon Blanke
Email: simon.blanke@yahoo.com
License: MIT License
"""

import importlib.metadata

__version__ = importlib.metadata.version("hyperactive")
__license__ = "MIT"

# Re-export SearchSpace for convenient access
from hyperactive.search_space import SearchSpace

__all__ = ["SearchSpace"]
