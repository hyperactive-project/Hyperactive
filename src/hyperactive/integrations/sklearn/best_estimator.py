"""Best estimator utilities for scikit-learn integration.

Author: Simon Blanke
Email: simon.blanke@yahoo.com
License: MIT License
"""

from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._compat import _deprecate_xt_in_inverse_transform
from .utils import _estimator_has


# NOTE Implementations of following methods from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py
# Tag: 1.5.1
class BestEstimator:
    """BestEstimator class."""

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """
    Compute the log-likelihood of each sample.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    score : array-like of shape (n_samples,)
        Log-likelihood of each sample in X.
    """
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """
    Predict class labels for samples in X.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The predicted class labels.
    """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """
        Probability estimates.

    The returned estimates for all classes are ordered by the
    label of classes.

        Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    T : array-like of shape (n_samples, n_classes)
        Returns the probability of the sample for each class in the model.
    """
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """
    Predict class log-probabilities for samples in X.

    The returned log-estimates for all classes are ordered by the
    label of classes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    T : array-like of shape (n_samples, n_classes)
        Returns the log-probability of the sample for each class in the model.
    """
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """
    Predict confidence scores for samples.

    The confidence score for a sample is proportional to the signed
    distance of that sample to the hyperplane.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    scores : array-like of shape (n_samples,) or (n_samples, n_classes)
        Confidence scores per (sample, class) combination.
    """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """
    Transform X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    X_new : ndarray of shape (n_samples, n_out)
        Transformed array.
    """
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X=None, Xt=None):
        """
    Transform X back to its original space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    X_original : array-like of shape (n_samples, n_features)
        Original samples.
    """
        X = _deprecate_xt_in_inverse_transform(X, Xt)
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(X)

    @property
    def classes_(self):
        """
    The class labels classifier has known to the data.

    Returns
    -------
    classes : array-like of shape (n_classes,)
        Unique class labels known to the classifier.
    """
        _estimator_has("classes_")(self)
        return self.best_estimator_.classes_
