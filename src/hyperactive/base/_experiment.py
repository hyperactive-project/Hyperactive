"""Base class for experiment."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

import time

import numpy as np
from skbase.base import BaseObject

from hyperactive.base._history import SearchHistory


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

    def __init__(self):
        super().__init__()
        self._history = SearchHistory()

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

        start_time = time.perf_counter()
        res, metadata = self._evaluate(params)
        eval_time = time.perf_counter() - start_time

        res = np.float64(res)

        self._history.record(
            params=params,
            score=res,
            metadata=metadata,
            eval_time=eval_time,
        )

        return res, metadata

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

    @property
    def data(self) -> SearchHistory:
        """Access the collected data from optimization runs.

        Tracks all evaluations during optimization. Data accumulates across
        multiple optimization runs on the same experiment instance.

        Returns
        -------
        SearchHistory
            The data object with the following attributes and methods:

            Attributes:
            - ``history``: list[dict] - all recorded evaluations
            - ``n_trials``: int - total number of trials
            - ``n_runs``: int - number of optimization runs
            - ``best_trial``: dict | None - trial with highest score
            - ``best_score``: float | None - highest score
            - ``best_params``: dict | None - parameters of best trial

            Methods:
            - ``get_run(run_id)``: get trials from specific run
            - ``clear()``: reset all data
            - ``new_run()``: signal start of new run (call before each run)

        Examples
        --------
        >>> experiment.data.history  # all evaluations as list of dicts
        >>> experiment.data.best_score  # highest score
        >>> experiment.data.get_run(0)  # evaluations from first run
        >>> experiment.data.clear()  # reset data

        To convert to a pandas DataFrame::

            import pandas as pd
            df = pd.DataFrame(experiment.data.history)
        """
        return self._history
