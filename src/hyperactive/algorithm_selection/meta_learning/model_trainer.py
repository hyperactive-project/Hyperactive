"""Model training for meta-learning pipeline.

This module trains a Random Forest classifier for pairwise optimizer ranking.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split


class RankingModelTrainer:
    """Train Random Forest model for pairwise optimizer ranking.

    This class trains a binary classifier that predicts which of two
    optimizers will perform better on a given problem.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=10
        Maximum depth of trees. None for unlimited.
    min_samples_leaf : int, default=5
        Minimum samples required at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all cores).

    Attributes
    ----------
    model : RandomForestClassifier
        The trained model.
    feature_importances_ : np.ndarray
        Feature importance scores after training.

    Examples
    --------
    >>> trainer = RankingModelTrainer(n_estimators=100)
    >>> trainer.train(X_train, y_train)
    >>> metrics = trainer.evaluate(X_test, y_test)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.feature_importances_: Optional[np.ndarray] = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the ranking model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Binary labels of shape (n_samples,).
        """
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pairwise comparison outcomes.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for pairwise comparisons.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability of class 1 (optimizer A wins).
        """
        return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate model performance.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test labels.

        Returns
        -------
        dict
            Dictionary with evaluation metrics:
            - accuracy: Classification accuracy
            - precision: Precision for class 1
            - recall: Recall for class 1
            - f1: F1 score
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        y_pred = self.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> dict[str, float]:
        """Perform cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        dict
            Dictionary with mean and std of accuracy across folds.
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        return {
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
        }

    def get_top_features(
        self, feature_names: list[str], n_top: int = 20
    ) -> list[tuple[str, float]]:
        """Get top features by importance.

        Parameters
        ----------
        feature_names : list of str
            Names of features.
        n_top : int, default=20
            Number of top features to return.

        Returns
        -------
        list of tuple
            List of (feature_name, importance) tuples.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained yet")

        indices = np.argsort(self.feature_importances_)[::-1][:n_top]
        return [
            (feature_names[i], float(self.feature_importances_[i]))
            for i in indices
        ]

    def save_model(self, path: str):
        """Save trained model using joblib.

        Parameters
        ----------
        path : str
            Output file path (should end with .joblib).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load trained model from file.

        Parameters
        ----------
        path : str
            Input file path.
        """
        self.model = joblib.load(path)
        self.feature_importances_ = self.model.feature_importances_


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[RankingModelTrainer, dict[str, float]]:
    """Convenience function to train and evaluate a ranking model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    test_size : float, default=0.2
        Fraction of data for testing.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Whether to print results.

    Returns
    -------
    trainer : RankingModelTrainer
        Trained model.
    metrics : dict
        Evaluation metrics on test set.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class balance: {np.mean(y_train):.3f} (train), {np.mean(y_test):.3f} (test)")

    # Train model
    trainer = RankingModelTrainer(random_state=random_state)
    trainer.train(X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    if verbose:
        print(f"\nTest set metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")

    return trainer, metrics
