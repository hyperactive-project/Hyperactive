"""Optimizer ranking model wrapper for the algorithm selector.

This module provides a simple interface for the AlgorithmSelector
to use the trained ML model for ranking optimizers.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .model import AlgorithmSelectionModel


class OptimizerRankingModel:
    """Static wrapper for optimizer ranking using the trained model.

    This class provides class methods for the AlgorithmSelector to
    easily check for and use the trained ML model.
    """

    _model: Optional[AlgorithmSelectionModel] = None

    @classmethod
    def get_model_path(cls) -> Path:
        """Get the path to the pre-trained model.

        Returns
        -------
        Path
            Path to the model file.
        """
        return AlgorithmSelectionModel.DEFAULT_MODEL_PATH

    @classmethod
    def is_model_available(cls) -> bool:
        """Check if a pre-trained model is available.

        Returns
        -------
        bool
            True if model file exists.
        """
        return cls.get_model_path().exists()

    @classmethod
    def _load_model(cls) -> Optional[AlgorithmSelectionModel]:
        """Load the model (cached after first load).

        Returns
        -------
        AlgorithmSelectionModel or None
            The loaded model, or None if unavailable.
        """
        if cls._model is not None:
            return cls._model

        if not cls.is_model_available():
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls._model = AlgorithmSelectionModel.load()
            return cls._model
        except Exception:
            return None

    @classmethod
    def rank_optimizers(
        cls,
        problem_features: list[float],
        ss_features_dict: Optional[dict] = None,
        ast_features_dict: Optional[dict] = None,
        n_iter: int = 100,
    ) -> dict[str, float]:
        """Rank optimizers for a problem described by features.

        Parameters
        ----------
        problem_features : list of float
            Combined feature vector (used for backwards compatibility).
        ss_features_dict : dict, optional
            Search space features as dict (preferred).
        ast_features_dict : dict, optional
            AST features as dict. Keys should match ASTFeatures.feature_names().
        n_iter : int, default=100
            Number of iterations.

        Returns
        -------
        dict
            Optimizer name to score mapping (sorted descending).
        """
        model = cls._load_model()
        if model is None:
            return {}

        try:
            # Build model input matching training format
            # Base features: search space features (10) + n_iter (1)
            # Then AST features (35) if model was trained with them

            if ss_features_dict is not None:
                # Use provided search space features
                model_input = [
                    ss_features_dict.get("n_dimensions", 2),
                    ss_features_dict.get("total_size", 100),
                    ss_features_dict.get("n_continuous", 0),
                    ss_features_dict.get("n_discrete", 2),
                    ss_features_dict.get("n_categorical", 0),
                    ss_features_dict.get("avg_choices_per_dim", 10),
                    ss_features_dict.get("min_choices", 10),
                    ss_features_dict.get("max_choices", 10),
                    ss_features_dict.get("avg_range_span", 10),
                    1.0 if ss_features_dict.get("has_mixed_types", False) else 0.0,
                    n_iter,
                ]

                # Add AST features if provided
                if ast_features_dict is not None:
                    from ..ast_feature_engineering import ASTFeatures

                    for name in ASTFeatures.feature_names():
                        model_input.append(float(ast_features_dict.get(name, 0)))

                model_input = np.array(model_input, dtype=np.float32)
            else:
                # Use combined vector directly
                model_input = np.array(problem_features, dtype=np.float32)

            # Ensure correct size for model
            expected = len(model.feature_names_)
            if len(model_input) < expected:
                model_input = np.pad(model_input, (0, expected - len(model_input)))
            elif len(model_input) > expected:
                model_input = model_input[:expected]

            return model.rank(model_input)

        except Exception:
            return {}

    @classmethod
    def clear_cache(cls):
        """Clear the cached model."""
        cls._model = None
