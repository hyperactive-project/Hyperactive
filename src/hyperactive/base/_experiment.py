"""Base class for experiment."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
from skbase.base import BaseObject


class BaseExperiment(BaseObject):
    """Base class for experiment."""

    _tags = {
        "object_type": "experiment",
        "python_dependencies": None,
        "property:randomness": "random",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
        "property:higher_or_lower_is_better": "higher",  # "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    _config = {
        "callbacks_pre": None,
        "callbacks_post": None,
    }

    def __init__(self):
        super().__init__()

    def __call__(self, params):
        """Score parameters. Same as score call, returns only a first element."""
        score, _ = self.score(params)
        return score

    @property
    def __name__(self):
        return type(self).__name__

    def paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str, or None
            The parameter names of the search parameters.

            * If list of str, params in ``evaluate`` and ``score`` must match this list,
              or a subset thereof.
            * If None, arbitrary parameters can be passed to ``evaluate`` and ``score``.
        """
        return self._paramnames()

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str, or None
            The parameter names of the search parameters.
            If not known or arbitrary, return None.
        """
        return None

    def evaluate(self, params):
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
        paramnames = self.paramnames()
        if paramnames is not None and not set(params.keys()) <= set(paramnames):
            raise ValueError(
                f"Parameters passed to {type(self)}.evaluate do not match: "
                f"expected {paramnames}, got {list(params.keys())}."
            )

        self._run_callbacks_pre(params)
        res, metadata = self._evaluate(params)
        res = np.float64(res)
        self._run_callbacks_post(params, res, metadata)

        return res, metadata

    def _run_callbacks_pre(self, params):
        """Run pre-evaluation callbacks."""
        callbacks = self.get_config().get("callbacks_pre")
        if callbacks:
            for callback in callbacks:
                callback(self, params)

    def _run_callbacks_post(self, params, result, metadata):
        """Run post-evaluation callbacks."""
        callbacks = self.get_config().get("callbacks_post")
        if callbacks:
            for callback in callbacks:
                callback(self, params, result, metadata)

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
        raise NotImplementedError

    def score(self, params):
        """Score the parameters - with sign such that higher is always better.

        Same as ``evaluate`` call except for the sign chosen so that higher is better.

        If the tag ``property:higher_or_lower_is_better`` is set to
        ``"lower"``, the result is ``-self.evaluate(params)``.

        If the tag is set to ``"higher"``, the result is
        identical to ``self.evaluate(params)``.

        Parameters
        ----------
        params : dict with string keys
            Parameters to score.

        Returns
        -------
        float
            The score of the parameters.
        dict
            Additional metadata about the search.
        """
        hib = self.get_tag("property:higher_or_lower_is_better", "lower")
        if hib == "higher":
            sign = 1
        elif hib == "lower":
            sign = -1
        elif hib == "mixed":
            raise NotImplementedError(
                "Score is undefined for mixed objectives. Override `score` or "
                "set a concrete objective where higher or lower is better."
            )
        else:
            raise ValueError(
                f"Unknown value for tag 'property:higher_or_lower_is_better': {hib}"
            )

        eval_res = self.evaluate(params)
        value = eval_res[0]
        metadata = eval_res[1]

        return sign * value, metadata
