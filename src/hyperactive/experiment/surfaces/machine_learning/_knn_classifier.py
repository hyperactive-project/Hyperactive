"""K-Nearest Neighbors Classifier test function wrapper."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

from hyperactive.experiment.surfaces.machine_learning._base import (
    BaseMachineLearningExperiment,
)


class KNeighborsClassifier(BaseMachineLearningExperiment):
    """K-Nearest Neighbors Classifier test function.

    A machine learning test function that evaluates K-Nearest Neighbors
    classification with different hyperparameters using cross-validation.

    This function optimizes the following hyperparameters:
        - n_neighbors: Number of neighbors to use
        - algorithm: Algorithm used to compute nearest neighbors
        - cv: Number of cross-validation folds
        - dataset: The dataset to evaluate on

    Parameters
    ----------
    metric : str, default="accuracy"
        Scoring metric for cross-validation.
        Common values: "accuracy", "f1", "precision", "recall".

    Attributes
    ----------
    para_names : list
        Names of the hyperparameters: n_neighbors, algorithm, cv, dataset.

    See Also
    --------
    surfaces.test_functions.machine_learning.KNeighborsClassifierFunction :
        The underlying Surfaces implementation.
    sklearn.neighbors.KNeighborsClassifier :
        The scikit-learn classifier being optimized.

    Notes
    -----
    The default search space includes three datasets from scikit-learn:
    digits, wine, and iris. The ``dataset`` parameter expects a callable
    that returns ``(X, y)`` when called.

    Examples
    --------
    Basic evaluation:

    >>> from hyperactive.experiment.surfaces import KNeighborsClassifier
    >>> func = KNeighborsClassifier(metric="accuracy")
    >>> space = func.get_default_search_space()
    >>> params = {
    ...     "n_neighbors": 5,
    ...     "algorithm": "auto",
    ...     "cv": 3,
    ...     "dataset": space["dataset"][0]  # digits dataset
    ... }
    >>> score, metadata = func.score(params)

    Optimization with Hyperactive:

    >>> from hyperactive.experiment.surfaces import KNeighborsClassifier
    >>> from hyperactive.opt.gfo import RandomSearch
    >>>
    >>> func = KNeighborsClassifier(metric="accuracy")
    >>> space = func.get_default_search_space()
    >>>
    >>> # Use a subset of the search space for faster optimization
    >>> search_space = {
    ...     "n_neighbors": [3, 5, 7, 9, 11, 15, 20],
    ...     "algorithm": ["auto", "ball_tree", "kd_tree"],
    ...     "cv": [3, 5],
    ...     "dataset": space["dataset"],  # digits, wine, iris
    ... }
    >>>
    >>> optimizer = RandomSearch(
    ...     search_space=search_space,
    ...     n_iter=20,
    ...     experiment=func,
    ... )
    >>> best_params = optimizer.solve()  # doctest: +SKIP
    """

    _tags = {
        "object_type": "experiment",
        "property:randomness": "random",
        "property:higher_or_lower_is_better": "higher",
        "property:function_family": "surfaces",
        "property:domain": "machine_learning",
        "property:task_type": "classification",
        "property:model_family": "neighbors",
        "python_dependencies": "scikit-learn",
    }

    _surfaces_class = KNeighborsClassifierFunction

    def __init__(self, metric="accuracy"):
        self.metric = metric
        super().__init__(metric=metric)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return [{"metric": "accuracy"}, {"metric": "f1_macro"}]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        # Import here to avoid import errors if surfaces not installed
        from surfaces.test_functions.machine_learning.tabular.classification.datasets import (
            iris_data,
        )

        return [
            {
                "n_neighbors": 5,
                "algorithm": "auto",
                "cv": 2,
                "dataset": iris_data,
            },
            {
                "n_neighbors": 3,
                "algorithm": "ball_tree",
                "cv": 3,
                "dataset": iris_data,
            },
        ]
