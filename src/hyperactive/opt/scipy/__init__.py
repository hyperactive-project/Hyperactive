"""Scipy optimization backend for Hyperactive.

This module provides optimizers from scipy.optimize for continuous
parameter optimization.

Note: Scipy optimizers only support continuous parameter spaces (tuples).
For discrete or categorical parameters, use optuna or gfo backends.
"""

from ._basinhopping import ScipyBasinhopping
from ._differential_evolution import ScipyDifferentialEvolution
from ._direct import ScipyDirect
from ._dual_annealing import ScipyDualAnnealing
from ._nelder_mead import ScipyNelderMead
from ._powell import ScipyPowell
from ._shgo import ScipySHGO

__all__ = [
    "ScipyBasinhopping",
    "ScipyDifferentialEvolution",
    "ScipyDirect",
    "ScipyDualAnnealing",
    "ScipyNelderMead",
    "ScipyPowell",
    "ScipySHGO",
]
