import numpy as np
import pytest
from hyperactive.opt.lipo import LIPOOptimizer


def sphere(params):
    return -(params["x"] ** 2 + params["y"] ** 2)


def test_lipo_basic():
    opt = LIPOOptimizer(
        search_space={
            "x": np.arange(-5, 5, 0.1),
            "y": np.arange(-5, 5, 0.1),
        },
        n_iter=50,
        experiment=sphere,
    )
    best = opt.solve()
    assert "x" in best and "y" in best
    assert abs(best["x"]) < 1.5   # should be near 0


def test_lipo_categorical():
    def fn(p): return 1.0 if p["kernel"] == "rbf" else 0.0
    opt = LIPOOptimizer(
        search_space={"kernel": ["linear", "rbf", "poly"]},
        n_iter=20, experiment=fn,
    )
    best = opt.solve()
    assert best["kernel"] == "rbf"


def test_lipo_snap_to_grid():
    def fn(p): return -abs(p["x"] - 3)
    opt = LIPOOptimizer(
        search_space={"x": np.array([1, 2, 3, 4, 5])},
        n_iter=30, experiment=fn,
    )
    best = opt.solve()
    assert best["x"] in [1, 2, 3, 4, 5]  # must be on the grid