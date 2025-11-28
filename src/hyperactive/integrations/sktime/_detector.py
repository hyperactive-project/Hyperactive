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
    ):
        self.detector = detector
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params
        super().__init__()

    def _fit(self, X, y):
        detector = self.detector.clone()

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
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
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

        params_default = {
            "detector": DummyDetector() if DummyDetector is not None else None,
            "optimizer": GridSearchSk(param_grid={}),
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
        }

        if parameter_set == "default":
            return params_default
        elif parameter_set == "more_params":
            return params_more
        else:
            return params_default
