"""Best estimator utilities for scikit-learn integration.

Author: Simon Blanke
Email: simon.blanke@yahoo.com
License: MIT License
"""

from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._compat import _deprecate_Xt_in_inverse_transform
from .utils import _estimator_has


# NOTE Implementations of following methods from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py
# Tag: 1.5.1
class BestEstimator:
    """BestEstimator class."""

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            The ``best_estimator_.score_samples`` method.
        """
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
                or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X=None, Xt=None):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        X_original : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Result of the `inverse_transform` function for `X` based on the
            estimator with the best found parameters.
        """
        X = _deprecate_Xt_in_inverse_transform(X, Xt)
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(X)

    @property
    def classes_(self):
        """Classes  function."""
        _estimator_has("classes_")(self)
        return self.best_estimator_.classes_
