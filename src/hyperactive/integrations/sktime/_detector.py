import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("sktime", severity="none"):
    # try to import a delegated detector base if present in sktime
    try:
        from sktime.annotation._delegate import _DelegatedDetector
    except Exception:
        from skbase.base import BaseEstimator as _DelegatedDetector
else:
    from skbase.base import BaseEstimator as _DelegatedDetector

from hyperactive.experiment.integrations.sktime_detector import (
    SktimeDetectorExperiment,
)


class TSDetectorOptCv(_DelegatedDetector):
    """
    Tune an sktime detector via any optimizer in the hyperactive toolbox.

    This mirrors the interface of other sktime wrappers in this package and
    delegates the tuning work to `SktimeDetectorExperiment`.
    """

    _tags = {
        "authors": "arnavk23",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",
        "object_type": "optimizer",
    }

    _delegate_name = "best_detector_"

    def __init__(
        self,
        detector,
        optimizer,
        cv=None,
        scoring=None,
        refit=True,
        error_score=np.nan,
        backend=None,
        backend_params=None,
        experiment=None,
    ):
        self.detector = detector
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params
        self.experiment = experiment
        super().__init__()

    def solve(self):
        """
        Run the optimizer on the provided experiment and return best params.

        This mirrors the optimizer interface used elsewhere in Hyperactive
        so the skbase test harness can exercise the optimizer without
        fitting data. The experiment must be supplied at construction time.
        """
        if self.experiment is None:
            raise ValueError(
                "TSDetectorOptCv requires an experiment instance to solve."
            )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=self.experiment)
        best_params = optimizer.solve()
        # If no detector was supplied (soft dependency missing), return
        if self.detector is None:
            self.best_params_ = best_params
            self.best_detector_ = None
            return best_params

        detector = self.detector.clone()
        self.best_params_ = best_params
        self.best_detector_ = detector.set_params(**best_params)
        return best_params

    def _fit(self, X, y):
        detector = self.detector.clone()

        # If an experiment was provided at construction time, use it.
        # Otherwise create a fresh SktimeDetectorExperiment for this fit.
        if getattr(self, "experiment", None) is not None:
            experiment = self.experiment
        else:
            experiment = SktimeDetectorExperiment(
                detector=detector,
                X=X,
                y=y,
                scoring=self.scoring,
                cv=self.cv,
                error_score=self.error_score,
                backend=self.backend,
                backend_params=self.backend_params,
            )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_detector_ = detector.set_params(**best_params)

        if self.refit:
            try:
                self.best_detector_.fit(X=X, y=y)
            except TypeError:
                self.best_detector_.fit(X=X)

        return self

    def _predict(self, X):
        if not self.refit:
            msg = (
                f"In {self.__class__.__name__}, refit must be True to make "
                f"predictions, but found refit=False. If refit=False, "
                f"{self.__class__.__name__} can be used only to tune "
                "hyper-parameters, as a parameter estimator."
            )
            raise RuntimeError(msg)
        return super()._predict(X=X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        if _check_soft_dependencies("sktime", severity="none"):
            try:
                from sktime.annotation.dummy import DummyDetector
            except Exception:
                DummyDetector = None
        else:
            DummyDetector = None

        from hyperactive.opt.gridsearch import GridSearchSk

        # Build a minimal experiment instance for test fixtures so that
        # optimizer test harnesses (which expect an `experiment` parameter)
        # can operate. If sktime deps are missing, fall back to None.
        X = None
        y = None
        test_experiment = None
        if DummyDetector is not None:
            try:
                from sktime.datasets import load_unit_test

                X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")
            except Exception:
                X = None
                y = None

        # Always try to build an experiment so optimizer tests have it
        try:
            if DummyDetector is not None:
                detector_instance = DummyDetector()
            else:
                detector_instance = None
            test_experiment = SktimeDetectorExperiment(
                detector=detector_instance,
                X=X,
                y=y,
            )
        except Exception:
            test_experiment = None

        params_default = {
            "detector": DummyDetector() if DummyDetector is not None else None,
            "optimizer": GridSearchSk(param_grid={}),
            "experiment": test_experiment,
        }

        params_more = {
            "detector": DummyDetector() if DummyDetector is not None else None,
            "optimizer": GridSearchSk(
                param_grid={"strategy": ["most_frequent", "stratified"]}
            ),
            "cv": 2,
            "scoring": None,
            "refit": False,
            "error_score": 0.0,
            "backend": "loky",
            "backend_params": {"n_jobs": 1},
            "experiment": test_experiment,
        }

        if parameter_set == "default":
            return params_default
        elif parameter_set == "more_params":
            return params_more
        else:
            return params_default
