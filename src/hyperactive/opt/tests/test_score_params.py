"""Tests for _score_params to guard against parameter passing regressions."""

import numpy as np
import pytest

from hyperactive.opt._common import _score_params


class _DictExperiment:
    """Minimal experiment stub that expects params as a single dict."""

    def __call__(self, params):
        return params["x"] ** 2 + params["y"] ** 2


class _DictOnlyExperiment:
    """Experiment stub that rejects keyword arguments.

    Fails loudly if params are passed as **kwargs instead of a dict,
    directly guarding against the ``experiment(**params)`` bug.
    """

    def __call__(self, params):
        if not isinstance(params, dict):
            raise TypeError(
                f"Expected a dict, got {type(params).__name__}. "
                "Parameters must be passed as a single dict, not as **kwargs."
            )
        return sum(v**2 for v in params.values())


def _make_meta(experiment, error_score=np.nan):
    return {"experiment": experiment, "error_score": error_score}


class TestScoreParams:
    """Tests for the _score_params helper function."""

    def test_params_passed_as_dict(self):
        """Params must be passed as a single dict, not unpacked as **kwargs."""
        exp = _DictOnlyExperiment()
        meta = _make_meta(exp)
        params = {"x": 3.0, "y": 4.0}

        score = _score_params(params, meta)

        assert score == 25.0

    def test_returns_correct_score(self):
        """Score must match the experiment's return value."""
        exp = _DictExperiment()
        meta = _make_meta(exp)

        assert _score_params({"x": 0.0, "y": 0.0}, meta) == 0.0
        assert _score_params({"x": 1.0, "y": 0.0}, meta) == 1.0
        assert _score_params({"x": 3.0, "y": 4.0}, meta) == 25.0

    def test_returns_python_float(self):
        """Return type must be a Python float, not numpy scalar."""
        exp = _DictExperiment()
        meta = _make_meta(exp)

        result = _score_params({"x": 1.0, "y": 1.0}, meta)
        assert type(result) is float

    def test_error_score_on_exception(self):
        """When the experiment raises, error_score must be returned."""

        def _failing_experiment(params):
            raise ValueError("intentional failure")

        meta = _make_meta(_failing_experiment, error_score=-999.0)

        with pytest.warns(match="intentional failure"):
            result = _score_params({"x": 1.0}, meta)

        assert result == -999.0

    def test_error_score_emits_warning(self):
        """A caught exception must produce a warning, never be silent."""

        def _failing_experiment(params):
            raise RuntimeError("boom")

        meta = _make_meta(_failing_experiment, error_score=np.nan)

        with pytest.warns(match="RuntimeError"):
            _score_params({"x": 1.0}, meta)

    def test_many_params_passed_as_dict(self):
        """Regression: many keys must not be unpacked as keyword arguments.

        With the old ``experiment(**params)`` bug, this would raise
        TypeError inside __call__ because it only accepts one argument.
        """

        def _sum_experiment(params):
            return sum(params.values())

        meta = _make_meta(_sum_experiment)
        params = {f"x{i}": float(i) for i in range(20)}

        score = _score_params(params, meta)

        assert score == float(sum(range(20)))

    def test_with_base_experiment(self):
        """Integration: works with a real BaseExperiment subclass."""
        from hyperactive.experiment.bench import Sphere

        exp = Sphere(n_dim=2)
        meta = _make_meta(exp)

        # Sphere minimum is at origin, value = 0
        # __call__ returns sign-adjusted score (higher is better)
        # Sphere is lower-is-better, so score = -evaluate
        score_origin = _score_params({"x0": 0.0, "x1": 0.0}, meta)
        score_away = _score_params({"x0": 3.0, "x1": 4.0}, meta)

        assert score_origin > score_away
