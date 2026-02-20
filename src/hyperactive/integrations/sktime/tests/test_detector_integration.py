"""Basic import smoke tests for the sktime detector integration."""


def test_detector_integration_imports():
    """Ensure the public integration symbols can be imported."""
    from hyperactive.experiment.integrations import SktimeDetectorExperiment
    from hyperactive.integrations.sktime import TSDetectorOptCv

    assert SktimeDetectorExperiment is not None
    assert TSDetectorOptCv is not None
