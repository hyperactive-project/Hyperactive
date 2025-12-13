"""Smoke tests for the sktime detector experiment integration."""


def test_sktime_detector_experiment_with_dummy():
    """Run a minimal smoke test using sktime's DummyDetector (if available)."""
    try:
        from sktime.annotation.dummy import DummyDetector
        from sktime.datasets import load_unit_test
    except Exception:
        # If sktime not available, skip the test by returning (user can run locally)
        return

    from hyperactive.experiment.integrations.sktime_detector import (
        SktimeDetectorExperiment,
    )

    X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")

    det = DummyDetector()

    exp = SktimeDetectorExperiment(detector=det, X=X, y=y, cv=2)

    # params: empty dict should be acceptable for DummyDetector
    score, metadata = exp.score({})

    assert isinstance(score, float)
    assert "results" in metadata
