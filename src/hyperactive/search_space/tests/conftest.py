"""Shared fixtures for SearchSpace tests."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest

from hyperactive.search_space import SearchSpace


@pytest.fixture
def simple_space():
    """Basic search space with all dimension types."""
    return SearchSpace(
        x=np.arange(-5, 5, 0.1),  # discrete
        y=(0.0, 10.0),  # continuous float
        lr=(1e-5, 1e-1, "log"),  # log-scale
        kernel=["rbf", "linear", "poly"],  # categorical
        seed=42,  # constant
    )


@pytest.fixture
def conditional_space():
    """Search space with conditions."""
    space = SearchSpace(
        kernel=["rbf", "linear", "poly"],
        C=(0.01, 100.0, "log"),
        gamma=(1e-4, 10.0, "log"),
        degree=[2, 3, 4, 5],
    )
    space.add_condition("gamma", when=lambda p: p["kernel"] != "linear")
    space.add_condition("degree", when=lambda p: p["kernel"] == "poly")
    return space


@pytest.fixture
def constrained_space():
    """Search space with constraints."""
    space = SearchSpace(
        batch_size=[32, 64, 128, 256],
        lr=(1e-5, 1e-1, "log"),
    )
    space.add_constraint(lambda p: p["batch_size"] * p["lr"] < 1.0)
    return space


def simple_objective(params):
    """Simple objective function for testing."""
    x = params.get("x", 0)
    y = params.get("y", 0)
    return -(x**2 + y**2)
