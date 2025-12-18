"""Surfaces test function wrappers for Hyperactive.

This module provides Hyperactive experiment wrappers for test functions
from the Surfaces package. These wrappers make Surfaces functions compatible
with Hyperactive's experiment interface and skbase tag system.

Mathematical Functions
----------------------
GramacyLee
    1D test function for surrogate modeling.
Ackley2D
    2D non-convex function with flat outer region.
Rastrigin
    N-dimensional highly multimodal function.

Machine Learning Functions
--------------------------
KNeighborsClassifier
    KNN classifier hyperparameter optimization.

Examples
--------
>>> from hyperactive.experiment.surfaces import Rastrigin
>>> func = Rastrigin(n_dim=5)
>>> params = {f"x{i}": 0.0 for i in range(5)}
>>> loss, _ = func.evaluate(params)

Notes
-----
The Surfaces package must be installed to use these experiments.
Install with: ``pip install surfaces``
"""

from hyperactive.experiment.surfaces.machine_learning import KNeighborsClassifier
from hyperactive.experiment.surfaces.mathematical import Ackley2D, GramacyLee, Rastrigin

__all__ = [
    # Mathematical functions
    "Ackley2D",
    "GramacyLee",
    "Rastrigin",
    # Machine learning functions
    "KNeighborsClassifier",
]
