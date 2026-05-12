"""Negation of experiment."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseExperiment


class Negative(BaseExperiment):
    """Negative of an experiment - flips the sign of the score.

    Can also be invoked by using the unary negation operator ``-``
    on an experiment instance, e.g., ``-experiment``.

    Useful in baselines or composite objectives.

    This composition class is configurable and allows to negate separately:

    * the sign of the score returned by the experiment,
    * the orientation of the optimization (minimization vs maximization).

    By default, both the score and the orientation are flipped,
    i.e., an experiment to maximize a function ``f(x)`` becomes an
    experiment to minimize ``-f(x)``, and vice versa.

    Parameters
    ----------
    experiment : BaseExperiment
        The experiment to be negated. It should be an instance of ``BaseExperiment``.

    flip_score : bool, default=True
        Whether to flip the score of the experiment. If True, the score will be
        negated, i.e., the score will be ``-f`` where ``f`` is the original score.

    flip_orientation : bool, default=True
        Whether to flip the orientation of the optimization. If True,
        minimization and maximization will be swapped in the experiment.

    Example
    -------
    >>> import numpy as np
    >>> from hyperactive.experiment.toy import Ackley
    >>> from hyperactive.experiment.compose import Negative
    >>>
    >>> ackley_exp = Ackley(a=20, b=0.2, c=2 * np.pi, d=2)
    >>> neg_ackley_exp = Negative(ackley_exp)
    """

    def __init__(self, experiment, flip_score=True, flip_orientation=True):
        self.experiment = experiment
        self.flip_score = flip_score
        self.flip_orientation = flip_orientation

        super().__init__()

        if self.flip_orientation:
            current_tag = self.get_tag("property:higher_or_lower_is_better", "mixed")
            if current_tag == "higher":
                self.set_tags(**{"property:higher_or_lower_is_better": "lower"})
            elif current_tag == "lower":
                self.set_tags(**{"property:higher_or_lower_is_better": "higher"})

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return self.experiment.paramnames()

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        value, metadata = self.experiment.evaluate(params)
        if self.flip_score:
            value = -value
        return value, metadata

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from hyperactive.experiment.toy import Ackley

        ackley_exp = Ackley(a=20, b=0.2, c=2, d=2)

        params0 = {"experiment": ackley_exp}
        params1 = {"experiment": ackley_exp, "flip_orientation": False}
        params2 = {"experiment": ackley_exp, "flip_score": False}

        return [params0, params1, params2]

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        params = {"x0": 0, "x1": 0}
        return [params, params, params]
