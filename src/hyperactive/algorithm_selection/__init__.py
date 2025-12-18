"""Algorithm selection for automatic optimizer recommendation.

This package provides tools for automatically selecting the best
optimization algorithm based on problem characteristics.

Main Classes
------------
AlgorithmSelector
    Analyzes objective function and search space to rank optimizers.
AutoOptimizer
    Drop-in optimizer that automatically selects the best algorithm.

Examples
--------
Using AlgorithmSelector to get recommendations:

>>> from hyperactive.algorithm_selection import AlgorithmSelector
>>> def objective(x):
...     return x["a"] ** 2 + x["b"] ** 2
>>> search_space = {"a": list(range(-5, 5)), "b": list(range(-5, 5))}
>>> selector = AlgorithmSelector()
>>> rankings = selector.rank(objective, search_space)

Using AutoOptimizer as a drop-in replacement:

>>> from hyperactive.algorithm_selection import AutoOptimizer
>>> auto = AutoOptimizer(
...     experiment=objective,
...     search_space=search_space,
...     n_iter=100,
... )
>>> best_params = auto.solve()
>>> auto.selected_optimizer_.__name__  # See which algorithm was chosen
'HillClimbing'
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from .auto_optimizer import AutoOptimizer
from .selector import AlgorithmSelector

__all__ = [
    "AlgorithmSelector",
    "AutoOptimizer",
]
