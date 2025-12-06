"""
Example: tune an sktime detector with Hyperactive's TSDetectorOptCv.

Run with:

    PYTHONPATH=src python examples/sktime_detector_example.py

This script uses a DummyDetector and a GridSearchSk optimizer as a minimal demo.
"""
from hyperactive.integrations.sktime import TSDetectorOptCv
from hyperactive.opt.gridsearch import GridSearchSk

try:
    from sktime.annotation.dummy import DummyDetector
    from sktime.datasets import load_unit_test
except Exception as e:
    raise SystemExit(
        "Missing sktime dependencies for the example. Install sktime to run this example."
    )


def main():
    X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")

    detector = DummyDetector()

    optimizer = GridSearchSk(param_grid={})

    tuned = TSDetectorOptCv(
        detector=detector,
        optimizer=optimizer,
        cv=2,
        refit=True,
    )

    tuned.fit(X=X, y=y)

    print("best_params:", tuned.best_params_)
    print("best_detector_:", tuned.best_detector_)


if __name__ == "__main__":
    main()
