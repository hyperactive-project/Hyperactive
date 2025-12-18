"""Base class for Surfaces machine learning test function wrappers."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.surfaces._base import BaseSurfacesExperiment


class BaseMachineLearningExperiment(BaseSurfacesExperiment):
    """Base class for wrapping Surfaces machine learning test functions.

    Machine learning test functions evaluate model performance based on
    hyperparameter configurations using cross-validation. They naturally
    return score values (higher is better) such as accuracy or R2.

    Subclasses must set:
        _surfaces_class : class
            The Surfaces ML function class to wrap.

    Parameters
    ----------
    metric : str, default="accuracy"
        Scoring metric for cross-validation. Common values:
        "accuracy", "r2", "f1", "precision", "recall".

    Notes
    -----
    ML functions naturally return score values (higher is better),
    so ``property:higher_or_lower_is_better`` is set to "higher".

    The ``property:randomness`` is set to "random" because ML evaluations
    involve cross-validation splits which introduce stochasticity.

    Examples
    --------
    >>> from hyperactive.experiment.surfaces import KNeighborsClassifier
    >>> func = KNeighborsClassifier(metric="accuracy")
    >>> space = func.get_default_search_space()
    >>> params = {
    ...     "n_neighbors": 5,
    ...     "algorithm": "auto",
    ...     "cv": 3,
    ...     "dataset": space["dataset"][0]
    ... }
    >>> score, _ = func.score(params)
    """

    _tags = {
        "object_type": "experiment",
        "property:randomness": "random",
        "property:higher_or_lower_is_better": "higher",
        "property:function_family": "surfaces",
        "property:domain": "machine_learning",
    }

    def __init__(self, metric="accuracy", **kwargs):
        super().__init__(metric=metric, **kwargs)

    def _evaluate(self, params):
        """Evaluate the parameters using the wrapped Surfaces function.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The score value from the Surfaces function (higher is better).
        dict
            Empty metadata dict.
        """
        # Use score() to get a consistent maximization value
        score = self._surfaces_func.score(params)
        return score, {}
