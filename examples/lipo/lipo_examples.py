import numpy as np
from hyperactive.opt.lipo import LIPOOptimizer


def objective(params):
    x, y = params["x"], params["y"]
    return -(x ** 2 + y ** 2)   # max at (0, 0)


opt = LIPOOptimizer(
    search_space={
        "x": np.arange(-5, 5, 0.1),
        "y": np.arange(-5, 5, 0.1),
    },
    n_iter=100,
    experiment=objective,
)
print(opt.solve())   # {'x': ~0.0, 'y': ~0.0}