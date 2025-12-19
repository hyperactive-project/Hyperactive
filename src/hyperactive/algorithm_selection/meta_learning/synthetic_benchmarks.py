"""Synthetic benchmark functions with diverse AST characteristics.

This module provides benchmark functions defined as pure Python code,
enabling AST feature extraction for meta-learning. Each function has
known characteristics that can be correlated with optimizer performance.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SyntheticBenchmark:
    """A synthetic benchmark function with metadata.

    Parameters
    ----------
    name : str
        Unique identifier for the benchmark.
    func : Callable
        The objective function.
    optimal_value : float
        Known optimal value (for evaluation).
    characteristics : dict
        Known problem characteristics.
    """

    name: str
    func: Callable
    optimal_value: float
    characteristics: dict


# =============================================================================
# Simple Unimodal Functions (favor local search)
# =============================================================================


def quadratic_2d(para):
    """Simple 2D quadratic function. Unimodal, smooth."""
    x = para["x0"]
    y = para["x1"]
    return -(x ** 2 + y ** 2)


def quadratic_5d(para):
    """5D quadratic function. Unimodal, smooth, higher dimensional."""
    total = 0
    for i in range(5):
        total += para[f"x{i}"] ** 2
    return -total


def quadratic_10d(para):
    """10D quadratic function. Unimodal, high dimensional."""
    total = 0
    for i in range(10):
        total += para[f"x{i}"] ** 2
    return -total


def shifted_quadratic(para):
    """Shifted quadratic - optimum not at origin."""
    x = para["x0"] - 2.5
    y = para["x1"] + 1.5
    return -(x ** 2 + y ** 2)


def scaled_quadratic(para):
    """Scaled quadratic - different scaling per dimension."""
    x = para["x0"]
    y = para["x1"]
    return -(0.5 * x ** 2 + 2.0 * y ** 2)


def linear_slope(para):
    """Linear function - trivial optimization."""
    x = para["x0"]
    y = para["x1"]
    return x + y


# =============================================================================
# Multimodal Functions (favor global search)
# =============================================================================


def sine_wave_2d(para):
    """Multimodal with sine waves. Many local optima."""
    x = para["x0"]
    y = para["x1"]
    return math.sin(x) + math.sin(y)


def cosine_wave_2d(para):
    """Multimodal with cosine waves."""
    x = para["x0"]
    y = para["x1"]
    return math.cos(x) + math.cos(y)


def sine_cosine_mix(para):
    """Mixed trig function - complex landscape."""
    x = para["x0"]
    y = para["x1"]
    return math.sin(x) * math.cos(y) + math.cos(x) * math.sin(y)


def rastrigin_like_2d(para):
    """Rastrigin-like function. Highly multimodal."""
    x = para["x0"]
    y = para["x1"]
    A = 10
    return -(A * 2 + (x ** 2 - A * math.cos(2 * math.pi * x)) +
             (y ** 2 - A * math.cos(2 * math.pi * y)))


def rastrigin_like_5d(para):
    """5D Rastrigin-like. Highly multimodal, challenging."""
    A = 10
    total = A * 5
    for i in range(5):
        xi = para[f"x{i}"]
        total += xi ** 2 - A * math.cos(2 * math.pi * xi)
    return -total


def schwefel_like(para):
    """Schwefel-like function. Global optimum far from local optima."""
    x = para["x0"]
    y = para["x1"]
    return -(418.9829 * 2 - x * math.sin(math.sqrt(abs(x))) -
             y * math.sin(math.sqrt(abs(y))))


def double_sine(para):
    """Double frequency sine waves."""
    x = para["x0"]
    y = para["x1"]
    return math.sin(2 * x) + math.sin(2 * y)


def triple_sine(para):
    """Triple frequency - more local optima."""
    x = para["x0"]
    y = para["x1"]
    return math.sin(3 * x) + math.sin(3 * y)


# =============================================================================
# Functions with Conditionals (discontinuous landscapes)
# =============================================================================


def step_function(para):
    """Step function - discontinuous."""
    x = para["x0"]
    y = para["x1"]
    if x > 0 and y > 0:
        return 1.0
    elif x > 0 or y > 0:
        return 0.5
    else:
        return 0.0


def piecewise_quadratic(para):
    """Piecewise quadratic with different regions."""
    x = para["x0"]
    y = para["x1"]
    if x < 0:
        return -(x ** 2 + y ** 2)
    else:
        return -((x - 1) ** 2 + y ** 2 + 0.5)


def threshold_function(para):
    """Function with threshold behavior."""
    x = para["x0"]
    y = para["x1"]
    base = -(x ** 2 + y ** 2)
    if base > -1:
        return base + 1
    else:
        return base


def multi_region(para):
    """Multiple distinct regions with different behaviors."""
    x = para["x0"]
    y = para["x1"]
    if x < -2:
        return -((x + 3) ** 2 + y ** 2)
    elif x > 2:
        return -((x - 3) ** 2 + y ** 2)
    else:
        return -(x ** 2 + y ** 2) + 1


def abs_function(para):
    """Absolute value - non-smooth at origin."""
    x = para["x0"]
    y = para["x1"]
    return -(abs(x) + abs(y))


# =============================================================================
# Functions with Loops (complex computation patterns)
# =============================================================================


def sum_of_powers(para):
    """Sum of different powers - loop pattern."""
    total = 0
    for i in range(5):
        xi = para[f"x{i}"]
        total += xi ** (i + 1)
    return -abs(total)


def weighted_sum(para):
    """Weighted sum with loop."""
    total = 0
    weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
    for i in range(5):
        total += weights[i] * para[f"x{i}"] ** 2
    return -total


def nested_computation(para):
    """Nested loop pattern."""
    total = 0
    for i in range(3):
        inner = 0
        for j in range(3):
            idx = i * 3 + j
            if idx < 5:
                inner += para[f"x{idx}"]
        total += inner ** 2
    return -total


# =============================================================================
# Functions with Math Operations (complex expressions)
# =============================================================================


def exponential_decay(para):
    """Exponential decay function."""
    x = para["x0"]
    y = para["x1"]
    return -math.exp(-(x ** 2 + y ** 2))


def log_barrier(para):
    """Logarithmic barrier function."""
    x = para["x0"]
    y = para["x1"]
    # Shift to ensure positive arguments
    return math.log(abs(x) + 1) + math.log(abs(y) + 1)


def sqrt_function(para):
    """Square root based function."""
    x = para["x0"]
    y = para["x1"]
    return -math.sqrt(x ** 2 + y ** 2 + 1)


def mixed_math(para):
    """Mix of different math operations."""
    x = para["x0"]
    y = para["x1"]
    return math.exp(-0.1 * (x ** 2 + y ** 2)) * math.cos(x) * math.cos(y)


def polynomial_high_degree(para):
    """Higher degree polynomial."""
    x = para["x0"]
    y = para["x1"]
    return -(x ** 4 + y ** 4 - 2 * x ** 2 - 2 * y ** 2)


# =============================================================================
# Benchmark Registry
# =============================================================================


def get_all_benchmarks() -> list[SyntheticBenchmark]:
    """Get all synthetic benchmark functions.

    Returns
    -------
    list of SyntheticBenchmark
        All available benchmark functions with metadata.
    """
    return [
        # Simple unimodal (local search friendly)
        SyntheticBenchmark(
            name="quadratic_2d",
            func=quadratic_2d,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 2, "smooth": True},
        ),
        SyntheticBenchmark(
            name="quadratic_5d",
            func=quadratic_5d,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 5, "smooth": True},
        ),
        SyntheticBenchmark(
            name="quadratic_10d",
            func=quadratic_10d,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 10, "smooth": True},
        ),
        SyntheticBenchmark(
            name="shifted_quadratic",
            func=shifted_quadratic,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 2, "smooth": True},
        ),
        SyntheticBenchmark(
            name="scaled_quadratic",
            func=scaled_quadratic,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 2, "smooth": True},
        ),
        SyntheticBenchmark(
            name="linear_slope",
            func=linear_slope,
            optimal_value=float("inf"),
            characteristics={"type": "linear", "dims": 2, "smooth": True},
        ),
        # Multimodal (global search friendly)
        SyntheticBenchmark(
            name="sine_wave_2d",
            func=sine_wave_2d,
            optimal_value=2.0,
            characteristics={"type": "multimodal", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="cosine_wave_2d",
            func=cosine_wave_2d,
            optimal_value=2.0,
            characteristics={"type": "multimodal", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="sine_cosine_mix",
            func=sine_cosine_mix,
            optimal_value=2.0,
            characteristics={"type": "multimodal", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="rastrigin_like_2d",
            func=rastrigin_like_2d,
            optimal_value=0.0,
            characteristics={"type": "highly_multimodal", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="rastrigin_like_5d",
            func=rastrigin_like_5d,
            optimal_value=0.0,
            characteristics={"type": "highly_multimodal", "dims": 5, "trig": True},
        ),
        SyntheticBenchmark(
            name="schwefel_like",
            func=schwefel_like,
            optimal_value=0.0,
            characteristics={"type": "deceptive", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="double_sine",
            func=double_sine,
            optimal_value=2.0,
            characteristics={"type": "multimodal", "dims": 2, "trig": True},
        ),
        SyntheticBenchmark(
            name="triple_sine",
            func=triple_sine,
            optimal_value=2.0,
            characteristics={"type": "highly_multimodal", "dims": 2, "trig": True},
        ),
        # Discontinuous (global search needed)
        SyntheticBenchmark(
            name="step_function",
            func=step_function,
            optimal_value=1.0,
            characteristics={"type": "discontinuous", "dims": 2, "conditionals": True},
        ),
        SyntheticBenchmark(
            name="piecewise_quadratic",
            func=piecewise_quadratic,
            optimal_value=0.0,
            characteristics={"type": "piecewise", "dims": 2, "conditionals": True},
        ),
        SyntheticBenchmark(
            name="threshold_function",
            func=threshold_function,
            optimal_value=1.0,
            characteristics={"type": "threshold", "dims": 2, "conditionals": True},
        ),
        SyntheticBenchmark(
            name="multi_region",
            func=multi_region,
            optimal_value=1.0,
            characteristics={"type": "multi_region", "dims": 2, "conditionals": True},
        ),
        SyntheticBenchmark(
            name="abs_function",
            func=abs_function,
            optimal_value=0.0,
            characteristics={"type": "non_smooth", "dims": 2, "conditionals": False},
        ),
        # Loop patterns
        SyntheticBenchmark(
            name="sum_of_powers",
            func=sum_of_powers,
            optimal_value=0.0,
            characteristics={"type": "polynomial", "dims": 5, "loops": True},
        ),
        SyntheticBenchmark(
            name="weighted_sum",
            func=weighted_sum,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 5, "loops": True},
        ),
        SyntheticBenchmark(
            name="nested_computation",
            func=nested_computation,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 5, "loops": True},
        ),
        # Complex math
        SyntheticBenchmark(
            name="exponential_decay",
            func=exponential_decay,
            optimal_value=0.0,
            characteristics={"type": "unimodal", "dims": 2, "exp": True},
        ),
        SyntheticBenchmark(
            name="log_barrier",
            func=log_barrier,
            optimal_value=0.0,
            characteristics={"type": "barrier", "dims": 2, "log": True},
        ),
        SyntheticBenchmark(
            name="sqrt_function",
            func=sqrt_function,
            optimal_value=-1.0,
            characteristics={"type": "unimodal", "dims": 2, "sqrt": True},
        ),
        SyntheticBenchmark(
            name="mixed_math",
            func=mixed_math,
            optimal_value=1.0,
            characteristics={"type": "multimodal", "dims": 2, "mixed": True},
        ),
        SyntheticBenchmark(
            name="polynomial_high_degree",
            func=polynomial_high_degree,
            optimal_value=2.0,
            characteristics={"type": "multimodal", "dims": 2, "polynomial": True},
        ),
    ]


def get_benchmarks_by_type(problem_type: str) -> list[SyntheticBenchmark]:
    """Get benchmarks filtered by problem type.

    Parameters
    ----------
    problem_type : str
        Type to filter by (e.g., "unimodal", "multimodal").

    Returns
    -------
    list of SyntheticBenchmark
        Filtered benchmarks.
    """
    all_benchmarks = get_all_benchmarks()
    return [b for b in all_benchmarks if b.characteristics.get("type") == problem_type]


def get_search_space_for_benchmark(benchmark: SyntheticBenchmark, resolution: int = 20) -> dict:
    """Get appropriate search space for a benchmark.

    Parameters
    ----------
    benchmark : SyntheticBenchmark
        The benchmark function.
    resolution : int, default=20
        Number of points per dimension.

    Returns
    -------
    dict
        Search space dictionary.
    """
    dims = benchmark.characteristics.get("dims", 2)
    return {f"x{i}": list(np.linspace(-5, 5, resolution)) for i in range(dims)}
