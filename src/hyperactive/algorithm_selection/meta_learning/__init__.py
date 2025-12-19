"""Meta-learning pipeline for algorithm selection.

This module provides tools for:
- Collecting benchmark data across optimizer × problem combinations
- Generating training datasets from collected data
- Training and serializing algorithm selection models
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from .data_collector import BenchmarkDataCollector, BenchmarkConfig, BenchmarkResult
from .dataset import DatasetGenerator
from .model import AlgorithmSelectionModel
from .ranking_model import OptimizerRankingModel

__all__ = [
    "BenchmarkDataCollector",
    "BenchmarkConfig",
    "BenchmarkResult",
    "DatasetGenerator",
    "AlgorithmSelectionModel",
    "OptimizerRankingModel",
]
