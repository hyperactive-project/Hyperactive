"""Search history for tracking optimization trials."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations


class SearchHistory:
    """Container for tracking optimization trial history.

    This class collects data from each evaluation during optimization runs.
    History accumulates across multiple optimization runs on the same experiment.

    Attributes
    ----------
    trials : list[dict]
        List of all recorded trials. Each trial is a dict containing:
        - iteration: int, global iteration index
        - run_id: int, which optimization run (0-indexed)
        - params: dict, the evaluated parameters
        - score: float, the evaluation score (raw, not sign-corrected)
        - metadata: dict, additional metadata from the experiment
        - eval_time: float, evaluation time in seconds
    """

    def __init__(self):
        self._trials: list[dict] = []
        self._current_run_id: int = 0

    def record(
        self,
        params: dict,
        score: float,
        metadata: dict | None,
        eval_time: float,
    ) -> None:
        """Record a single trial.

        Parameters
        ----------
        params : dict
            The evaluated parameters.
        score : float
            The evaluation score (raw, not sign-corrected).
        metadata : dict or None
            Additional metadata from the experiment.
        eval_time : float
            Evaluation time in seconds.
        """
        self._trials.append({
            "iteration": len(self._trials),
            "run_id": self._current_run_id,
            "params": dict(params),
            "score": float(score),
            "metadata": dict(metadata) if metadata else {},
            "eval_time": float(eval_time),
        })

    def new_run(self) -> None:
        """Signal the start of a new optimization run.

        Increments the run_id counter. Subsequent trials will be tagged
        with the new run_id.
        """
        self._current_run_id += 1

    def clear(self) -> None:
        """Clear all history data and reset run counter."""
        self._trials = []
        self._current_run_id = 0

    @property
    def history(self) -> list[dict]:
        """Return all recorded evaluations as a list.

        Returns
        -------
        list[dict]
            List of all evaluations. Each entry contains iteration, run_id,
            params, score, metadata, and eval_time.
        """
        return self._trials

    @property
    def n_trials(self) -> int:
        """Return the total number of recorded trials.

        Returns
        -------
        int
            Number of trials across all runs.
        """
        return len(self._trials)

    @property
    def n_runs(self) -> int:
        """Return the number of optimization runs.

        Returns
        -------
        int
            Number of runs (0 if no trials recorded yet).
        """
        return self._current_run_id + 1

    @property
    def best_trial(self) -> dict | None:
        """Return the trial with the highest score.

        Returns
        -------
        dict or None
            The trial dict with the highest score, or None if no trials.
        """
        if not self._trials:
            return None
        return max(self._trials, key=lambda t: t["score"])

    @property
    def best_score(self) -> float | None:
        """Return the highest score across all trials.

        Returns
        -------
        float or None
            The highest score, or None if no trials.
        """
        best = self.best_trial
        return best["score"] if best else None

    @property
    def best_params(self) -> dict | None:
        """Return the parameters of the best trial.

        Returns
        -------
        dict or None
            Parameters of the trial with highest score, or None if no trials.
        """
        best = self.best_trial
        return best["params"] if best else None

    def get_run(self, run_id: int) -> list[dict]:
        """Return all trials from a specific run.

        Parameters
        ----------
        run_id : int
            The run identifier (0-indexed).

        Returns
        -------
        list[dict]
            List of trials from the specified run.
        """
        return [t for t in self._trials if t["run_id"] == run_id]

    def __len__(self) -> int:
        """Return the number of trials."""
        return len(self._trials)

    def __repr__(self) -> str:
        return f"SearchHistory(n_trials={self.n_trials}, n_runs={self.n_runs})"
