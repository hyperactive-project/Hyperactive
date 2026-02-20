"""Common functions used by multiple optimizers."""

import warnings

__all__ = ["_score_params"]


def _score_params(params, meta):
    """Score parameters, used in parallelization.

    Uses experiment.score (via __call__), which is standardized to
    "higher-is-better" across experiments.
    """
    meta = meta.copy()
    experiment = meta["experiment"]
    error_score = meta["error_score"]

    try:
        return float(experiment(params))
    except Exception as e:
        warnings.warn(
            f"Experiment raised {type(e).__name__}: {e}. "
            f"Assigning error_score={error_score}.",
            stacklevel=2,
        )
        return float(error_score)
