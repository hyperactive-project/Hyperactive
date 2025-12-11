"""Pytest configuration for documentation snippets.

This conftest provides shared fixtures that snippet files can use for testing.
The fixtures ensure consistent behavior across all snippet tests.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_search_space():
    """Simple search space for basic examples."""
    return {
        "x": np.arange(-5, 5, 0.1),
        "y": np.arange(-5, 5, 0.1),
    }


@pytest.fixture
def simple_objective():
    """Simple objective function for basic examples."""

    def objective(params):
        x = params["x"]
        y = params["y"]
        return -(x**2 + y**2)

    return objective


@pytest.fixture
def sklearn_data():
    """Load iris dataset for sklearn examples."""
    from sklearn.datasets import load_iris

    return load_iris(return_X_y=True)


@pytest.fixture
def sklearn_train_test_split(sklearn_data):
    """Split sklearn data into train and test sets."""
    from sklearn.model_selection import train_test_split

    X, y = sklearn_data
    return train_test_split(X, y, test_size=0.2, random_state=42)
