"""Algorithm selection model training and inference.

This module provides the AlgorithmSelectionModel class for training
and using ML models to predict the best optimization algorithm.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np


class AlgorithmSelectionModel:
    """ML model for algorithm selection.

    This class wraps a scikit-learn classifier to predict which
    optimization algorithm will perform best on a given problem.

    Parameters
    ----------
    model_type : str, default="random_forest"
        Type of model to use. Options: "random_forest", "gradient_boosting".

    Attributes
    ----------
    model_ : sklearn estimator
        The trained model.
    optimizer_names_ : list of str
        Names of optimizers (class labels).
    feature_names_ : list of str
        Names of input features.
    is_fitted_ : bool
        Whether the model has been trained.

    Examples
    --------
    >>> model = AlgorithmSelectionModel()
    >>> model.fit(X_train, y_train, feature_names, optimizer_names)
    >>> predictions = model.predict(X_test)
    >>> scores = model.predict_proba(X_test)
    """

    # Default path for pre-trained model
    DEFAULT_MODEL_PATH = Path(__file__).parent / "pretrained_model.pkl"

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in ("random_forest", "gradient_boosting"):
            raise ValueError(
                "model_type must be 'random_forest' or 'gradient_boosting'"
            )
        self.model_type = model_type

        self.model_ = None
        self.optimizer_names_: Optional[list[str]] = None
        self.feature_names_: Optional[list[str]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        optimizer_names: list[str],
        verbose: bool = False,
    ) -> "AlgorithmSelectionModel":
        """Train the algorithm selection model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label array of shape (n_samples,) with optimizer indices.
        feature_names : list of str
            Names of input features.
        optimizer_names : list of str
            Names of optimizers (for decoding predictions).
        verbose : bool, default=False
            Whether to print training progress.

        Returns
        -------
        self
            The fitted model.
        """
        # Suppress warnings during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier

                self.model_ = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                )
            else:  # gradient_boosting
                from sklearn.ensemble import GradientBoostingClassifier

                self.model_ = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=0,
                )

            self.model_.fit(X, y)

        self.feature_names_ = feature_names
        self.optimizer_names_ = optimizer_names
        self.is_fitted_ = True

        if verbose:
            print(f"Model trained on {len(y)} examples")
            print(f"  Features: {len(feature_names)}")
            print(f"  Optimizers: {len(optimizer_names)}")

        return self

    def predict(self, X: np.ndarray) -> list[str]:
        """Predict the best optimizer for each sample.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        list of str
            Predicted optimizer names.
        """
        self._check_fitted()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices = self.model_.predict(X)

        return [self.optimizer_names_[i] for i in indices]

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict probability scores for each optimizer.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dict
            Dictionary mapping optimizer names to probability arrays.
        """
        self._check_fitted()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.model_.predict_proba(X)

        # Map to optimizer names
        result = {}
        for i, name in enumerate(self.optimizer_names_):
            if i < proba.shape[1]:
                result[name] = proba[:, i]
            else:
                result[name] = np.zeros(proba.shape[0])

        return result

    def rank(self, features: np.ndarray) -> dict[str, float]:
        """Rank optimizers for a single problem.

        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        dict
            Dictionary mapping optimizer names to scores (sorted descending).
        """
        self._check_fitted()

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.model_.predict_proba(features)[0]

        # Create ranking dict
        scores = {}
        for i, name in enumerate(self.optimizer_names_):
            if i < len(proba):
                scores[name] = float(proba[i])
            else:
                scores[name] = 0.0

        # Sort by score descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the trained model to disk.

        Parameters
        ----------
        path : str or Path, optional
            Path to save the model. Defaults to DEFAULT_MODEL_PATH.

        Returns
        -------
        Path
            Path where model was saved.
        """
        self._check_fitted()

        if path is None:
            path = self.DEFAULT_MODEL_PATH
        path = Path(path)

        data = {
            "model": self.model_,
            "optimizer_names": self.optimizer_names_,
            "feature_names": self.feature_names_,
            "model_type": self.model_type,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        return path

    @classmethod
    def load(
        cls, path: Optional[Union[str, Path]] = None
    ) -> "AlgorithmSelectionModel":
        """Load a trained model from disk.

        Parameters
        ----------
        path : str or Path, optional
            Path to load the model from. Defaults to DEFAULT_MODEL_PATH.

        Returns
        -------
        AlgorithmSelectionModel
            The loaded model.
        """
        if path is None:
            path = cls.DEFAULT_MODEL_PATH
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"No pre-trained model found at {path}. "
                "Run the training pipeline first."
            )

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(model_type=data["model_type"])
        instance.model_ = data["model"]
        instance.optimizer_names_ = data["optimizer_names"]
        instance.feature_names_ = data["feature_names"]
        instance.is_fitted_ = True

        return instance

    @classmethod
    def load_or_train(
        cls,
        path: Optional[Union[str, Path]] = None,
        force_retrain: bool = False,
        verbose: bool = False,
    ) -> "AlgorithmSelectionModel":
        """Load pre-trained model or train a new one if not available.

        This is the recommended way to get a model for production use.

        Parameters
        ----------
        path : str or Path, optional
            Path to model file.
        force_retrain : bool, default=False
            If True, always retrain even if model exists.
        verbose : bool, default=False
            Whether to print progress during training.

        Returns
        -------
        AlgorithmSelectionModel
            The model (loaded or newly trained).
        """
        if path is None:
            path = cls.DEFAULT_MODEL_PATH
        path = Path(path)

        # Try to load existing model
        if not force_retrain and path.exists():
            try:
                return cls.load(path)
            except Exception:
                pass  # Fall through to training

        # Train new model
        if verbose:
            print("Training new algorithm selection model...")

        model = cls._train_default_model(verbose=verbose)
        model.save(path)

        return model

    @classmethod
    def _train_default_model(
        cls, verbose: bool = False
    ) -> "AlgorithmSelectionModel":
        """Train model on default benchmarks.

        Parameters
        ----------
        verbose : bool, default=False
            Whether to print progress.

        Returns
        -------
        AlgorithmSelectionModel
            Trained model.
        """
        from .data_collector import BenchmarkDataCollector
        from .dataset import DatasetGenerator

        # Collect benchmark data (reduced set for faster training)
        collector = BenchmarkDataCollector(n_runs=2, verbose=verbose)

        # Use subset of configs for faster training
        all_configs = collector.get_default_configs()

        # Sample configs: every 3rd to reduce training time
        configs = all_configs[::3]

        if verbose:
            print(f"Running {len(configs)} benchmark configurations...")

        results = collector.collect(configs)

        if verbose:
            print(f"Collected {len(results)} results")

        # Generate dataset
        generator = DatasetGenerator(aggregation="mean")
        X, y, feature_names, optimizer_names = generator.generate(
            results, configs
        )

        if verbose:
            print(f"Generated dataset: {X.shape[0]} examples, {X.shape[1]} features")

        # Train model
        model = cls(model_type="random_forest")
        model.fit(X, y, feature_names, optimizer_names, verbose=verbose)

        return model

    def _check_fitted(self):
        """Check if model has been fitted."""
        if not self.is_fitted_:
            raise ValueError(
                "Model has not been fitted. Call fit() or load() first."
            )

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importances from the trained model.

        Returns
        -------
        dict
            Feature name to importance score.
        """
        self._check_fitted()

        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
            return dict(
                sorted(
                    zip(self.feature_names_, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        return {}

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Evaluate the model on test data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.
        verbose : bool, default=False
            Whether to print evaluation results.

        Returns
        -------
        dict
            Dictionary with evaluation metrics.
        """
        self._check_fitted()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.metrics import accuracy_score, top_k_accuracy_score

            y_pred = self.model_.predict(X)

            metrics = {
                "accuracy": accuracy_score(y, y_pred),
            }

            # Top-k accuracy if we have probabilities and enough classes
            if hasattr(self.model_, "predict_proba"):
                y_proba = self.model_.predict_proba(X)
                n_classes = y_proba.shape[1]
                # Only compute top-k if we have more than 2 classes
                if n_classes > 2:
                    k = min(3, n_classes)
                    try:
                        metrics["top_k_accuracy"] = top_k_accuracy_score(
                            y, y_proba, k=k, labels=range(n_classes)
                        )
                    except ValueError:
                        pass  # Skip if there's an issue

        if verbose:
            print(f"Evaluation results:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.3f}")

        return metrics
