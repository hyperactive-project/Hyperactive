"""Built-in callbacks for experiments and optimizers."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

__all__ = [
    "HistoryCallback",
    "LoggingCallback",
    "SleepCallback",
    "TargetReachedCallback",
]

class HistoryCallback:
    """Records evaluation history.

    Note: Not thread-safe. If using parallel evaluations, history.append()
    may have race conditions.

    Attributes
    ----------
    history : list of dict
        List of evaluation records with keys: params, result, metadata.
    """

    def __init__(self):
        self.history = []

    def __call__(self, experiment, params, result, metadata):
        """Record an evaluation."""
        self.history.append(
            {
                "params": params.copy(),
                "result": result,
                "metadata": metadata.copy() if metadata else {},
            }
        )

    def clear(self):
        """Clear the recorded history."""
        self.history = []

    def get_best(self, higher_is_better=True):
        """Return the best evaluation record.

        Parameters
        ----------
        higher_is_better : bool, default=True
            If True, return highest result; if False, return lowest.

        Returns
        -------
        dict or None
            Best evaluation record, or None if history is empty.
        """
        if not self.history:
            return None
        key = (lambda x: x["result"]) if higher_is_better else (lambda x: -x["result"])
        return max(self.history, key=key)

class LoggingCallback:
    """Logs evaluations to console or a logger.

    Parameters
    ----------
    logger : logging.Logger or None, default=None
        Logger to use. If None, prints to stdout.
    level : str, default="info"
        Log level to use (only relevant if logger is provided).
    format_str : str or None, default=None
        Format string with placeholders: {params}, {result}, {metadata}, {count}.
    """

    def __init__(self, logger=None, level="info", format_str=None):
        self.logger = logger
        self.level = level
        self.format_str = format_str or "Eval {count}: {params} -> {result:.4f}"
        self._count = 0

    def __call__(self, experiment, params, result, metadata):
        """Log an evaluation."""
        self._count += 1
        msg = self.format_str.format(
            params=params,
            result=result,
            metadata=metadata,
            count=self._count,
        )
        if self.logger:
            getattr(self.logger, self.level)(msg)
        else:
            print(msg)

    def reset(self):
        """Reset the evaluation counter."""
        self._count = 0

class SleepCallback:
    """Adds a delay after each evaluation.

    Useful for simulating expensive evaluations during testing.

    Parameters
    ----------
    seconds : float
        Number of seconds to sleep after each evaluation.
    """

    def __init__(self, seconds):
        self.seconds = seconds

    def __call__(self, experiment, params, result, metadata):
        """Sleep after evaluation."""
        import time

        time.sleep(self.seconds)

class TargetReachedCallback:
    """Tracks if a target score has been reached.

    This callback only tracks whether the target was reached; it does not
    automatically stop the optimizer. Check the ``reached`` attribute to
    determine if optimization should be terminated.

    Parameters
    ----------
    target_score : float
        The target score to reach.
    higher_is_better : bool, default=True
        If True, target is reached when result >= target_score.
        If False, target is reached when result <= target_score.

    Attributes
    ----------
    reached : bool
        Whether the target score has been reached.
    best_result : float or None
        The best result seen so far.
    """

    def __init__(self, target_score, higher_is_better=True):
        self.target_score = target_score
        self.higher_is_better = higher_is_better
        self.reached = False
        self.best_result = None

    def __call__(self, experiment, params, result, metadata):
        """Check if target score is reached."""
        if self.best_result is None:
            self.best_result = result
        elif self.higher_is_better:
            self.best_result = max(self.best_result, result)
        else:
            self.best_result = min(self.best_result, result)

        if self.higher_is_better:
            self.reached = result >= self.target_score
        else:
            self.reached = result <= self.target_score

    def reset(self):
        """Reset the callback state."""
        self.reached = False
        self.best_result = None

