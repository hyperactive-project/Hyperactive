"""
Example: tune an sktime detector with Hyperactive's TSDetectorOptCv.

Run with:

    PYTHONPATH=src python examples/sktime_detector_example.py

This script uses a DummyRegularAnomalies detector and a GridSearchSk optimizer as a minimal demo.
"""
from hyperactive.integrations.sktime import TSDetectorOptCv
from hyperactive.opt.gridsearch import GridSearchSk

try:
    from sktime.detection.dummy import DummyRegularAnomalies
    from sktime.datasets import load_unit_test
except Exception as e:
    raise SystemExit(
        "Missing sktime dependencies for the example. Install sktime to run this example."
    )


def main():
    X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")

    detector = DummyRegularAnomalies()

    optimizer = GridSearchSk(param_grid={})

    # Create the tuning experiment using the TSDetectorOptCv wrapper
    # The wrapper requires an experiment instance and optimizer to solve
    from hyperactive.experiment.integrations.sktime_detector import (
        SktimeDetectorExperiment,
    )

    experiment = SktimeDetectorExperiment(
        detector=detector,
        X=X,
        y=y,
        cv=2,
    )

    tuned = TSDetectorOptCv(
        detector=detector,
        optimizer=optimizer,
        experiment=experiment,
    )

    best_params = tuned.solve()

    print("best_params:", best_params)
    print("best_detector_:", tuned.best_detector_)


if __name__ == "__main__":
    main()
