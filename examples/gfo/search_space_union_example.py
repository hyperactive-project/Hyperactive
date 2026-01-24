"""
Union Grid Search Space Example

Demonstrates GFO optimizers accepting:
- array-like values (lists/tuples coerced to numpy arrays)
- union grids (list of dicts)
"""

from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import GridSearch


def objective(params):
    return params["x"] + params["y"]


experiment = FunctionExperiment(objective)

# Array-like values are coerced internally (list + tuple shown here)
grid_a = {"x": [0, 1], "y": (0, 1)}

# Union grids: each dict is a separate grid; the optimizer runs all and keeps best
search_space = [
    grid_a,
    {"x": [2], "y": [3]},
]

optimizer = GridSearch(
    search_space=search_space,
    n_iter=5,
    experiment=experiment,
)

best_params = optimizer.solve()
print("Best params:", best_params)
