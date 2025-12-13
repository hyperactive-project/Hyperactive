"""Integration adapter for sktime detector experiments.

Provides `SktimeDetectorExperiment` which adapts sktime detector-style
objects to the Hyperactive experiment interface.
"""

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from hyperactive.base import BaseExperiment
from hyperactive.experiment.integrations._skl_metrics import _coerce_to_scorer_and_sign


class SktimeDetectorExperiment(BaseExperiment):
    """
    Experiment adapter for time series detector/anomaly detection experiments.

    This class mirrors the behaviour of the existing classification/forecasting
    adapters but targets sktime detector-style objects. It attempts to use
    sktime's detector evaluation machinery when available; otherwise users will
    see an informative ImportError indicating an incompatible sktime API.
    """

    _tags = {
        "authors": "arnavk23",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",
    }

    def __init__(
        self,
        detector,
        X,
        y,
        cv=None,
        scoring=None,
        error_score=np.nan,
        backend=None,
        backend_params=None,
    ):
        self.detector = detector
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        # use "classifier" as a safe default estimator type for metric coercion
        self._scoring, _sign = _coerce_to_scorer_and_sign(scoring, "classifier")

        _sign_str = "higher" if _sign == 1 else "lower"
        self.set_tags(**{"property:higher_or_lower_is_better": _sign_str})

        # default handling for cv similar to classification adapter
        if isinstance(cv, int):
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=cv, shuffle=True)
        elif cv is None:
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=3, shuffle=True)
        else:
            self._cv = cv

    def _paramnames(self):
        # If detector is not provided (e.g., soft-deps missing or test stub),
        # return an empty list so callers can still iterate safely.
        if self.detector is None:
            return []

        # Some detector-like objects may not expose `get_params`; guard that.
        try:
            return list(self.detector.get_params().keys())
        except Exception:
            return None

    def _evaluate(self, params):
        """
        Evaluate the parameters.

        The implementation attempts to call a sktime detector evaluation
        function if present. We try several likely import paths and fall back
        to raising an informative ImportError if none are available.
        """
        if self.detector is None:
            return np.nan, {}

        evaluate = None
        candidates = [
            "sktime.anomaly_detection.model_evaluation.evaluate",
            "sktime.detection.model_evaluation.evaluate",
            "sktime.annotation.model_evaluation.evaluate",
        ]

        for cand in candidates:
            mod_path, fn = cand.rsplit(".", 1)
            try:
                mod = __import__(mod_path, fromlist=[fn])
                evaluate = getattr(mod, fn)
                break
            except Exception:
                evaluate = None

        detector = self.detector.clone().set_params(**params)

        if evaluate is None:
            raise ImportError(
                "Could not find a compatible sktime detector evaluation function. "
                "Ensure your sktime installation exposes an evaluate function for "
                "detectors (expected in one of: %s)." % ", ".join(candidates)
            )

        # call the sktime evaluate function if available
        if evaluate is not None:
            results = evaluate(
                detector,
                cv=self._cv,
                X=self.X,
                y=self.y,
                scoring=getattr(self._scoring, "_metric_func", self._scoring),
                error_score=self.error_score,
                backend=self.backend,
                backend_params=self.backend_params,
            )

            metric = getattr(self._scoring, "_metric_func", self._scoring)
            result_name = f"test_{getattr(metric, '__name__', 'score')}"

            res_float = results[result_name].mean()

            return res_float, {"results": results}

        # Fallback: perform a manual cross-validation loop if `evaluate` is not present.

        # Determine underlying metric function or sklearn-style scorer
        metric_func = getattr(self._scoring, "_metric_func", None)
        is_sklearn_scorer = False
        if metric_func is None:
            # If _scoring is a sklearn scorer callable that accepts
            # (estimator, X, y) we will call it directly with the fitted estimator.
            if callable(self._scoring):
                # Heuristic: sklearn scorers produced by `make_scorer` take
                # arguments `(estimator, X, y)`.
                is_sklearn_scorer = True
        else:
            metric = metric_func

        scores = []
        # If X is None, try to build indices from y
        if self.X is None:
            for train_idx, test_idx in self._cv.split(self.y):
                X_train = None
                X_test = None
                if isinstance(self.y, list | tuple):
                    y_train = [self.y[i] for i in train_idx]
                    y_test = [self.y[i] for i in test_idx]
                else:
                    import numpy as _np

                    arr = _np.asarray(self.y)
                    y_train = arr[train_idx]
                    y_test = arr[test_idx]

                est = detector.clone().set_params(**params)
                try:
                    est.fit(X=None, y=y_train)
                except TypeError:
                    est.fit(X=None)

                try:
                    y_pred = est.predict(X=None)
                except TypeError:
                    y_pred = est.predict()

                if metric_func is not None:
                    score = metric_func(y_test, y_pred)
                elif is_sklearn_scorer:
                    score = self._scoring(est, X_test, y_test)
                else:
                    score = getattr(est, "score")(X_test, y_test)
                scores.append(score)
        else:
            for train_idx, test_idx in self._cv.split(self.X, self.y):
                X_train = self._safe_index(self.X, train_idx)
                X_test = self._safe_index(self.X, test_idx)
                y_train = self._safe_index(self.y, train_idx)
                y_test = self._safe_index(self.y, test_idx)

                est = detector.clone().set_params(**params)
                try:
                    est.fit(X=X_train, y=y_train)
                except TypeError:
                    est.fit(X=X_train)

                try:
                    y_pred = est.predict(X_test)
                except TypeError:
                    y_pred = est.predict()

                if metric_func is not None:
                    score = metric_func(y_test, y_pred)
                elif is_sklearn_scorer:
                    score = self._scoring(est, X_test, y_test)
                else:
                    score = getattr(est, "score")(X_test, y_test)

                scores.append(score)

        # average scores
        import numpy as _np

        res_float = _np.mean(scores)
        return float(res_float), {"results": {"cv_scores": scores}}

    def _safe_index(self, obj, idx):
        """Safely index into `obj` using integer indices.

        Supports pandas objects with ``.iloc``, numpy arrays/lists, and other
        indexable types.
        """
        try:
            return obj.iloc[idx]
        except Exception:
            try:
                import numpy as _np

                arr = _np.asarray(obj)
                return arr[idx]
            except Exception:
                return [obj[i] for i in idx]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        This returns a list or dict appropriate to construct test instances
        for this class. See the skbase test helpers for expected formats.
        """
        if _check_soft_dependencies("sktime", severity="none"):
            try:
                from sktime.annotation.dummy import DummyDetector
            except Exception:
                DummyDetector = None

            try:
                from sktime.datasets import load_unit_test

                X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")
            except Exception:
                X = None
                y = None
        else:
            DummyDetector = None
            X = None
            y = None

        params_default = {
            "detector": DummyDetector() if DummyDetector is not None else None,
            "X": X,
            "y": y,
        }

        params_more = {
            "detector": DummyDetector() if DummyDetector is not None else None,
            "X": X,
            "y": y,
            "cv": 2,
            "scoring": None,
            "error_score": 0.0,
            "backend": "loky",
            "backend_params": {"n_jobs": 1},
        }

        if parameter_set == "default":
            return [params_default]
        elif parameter_set == "more_params":
            return [params_more]
        else:
            return [params_default]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        The returned list should match the length of ``get_test_params()`` and
        contain dictionaries of hyperparameter settings that are valid
        inputs for ``score``/``evaluate`` when an instance is created from the
        corresponding element of ``get_test_params()``.
        """
        # For the simple detector tests, an empty dict of params is adequate.
        return [{}]
