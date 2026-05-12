"""Integration tests for sktime tuners."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("sktime", severity="none"):
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
    from sktime.split import ExpandingWindowSplitter

    from hyperactive.integrations.sktime import ForecastingOptCV, TSCOptCV
    from hyperactive.opt import GridSearchSk

    EST_TO_TEST = [ForecastingOptCV, TSCOptCV]
else:
    EST_TO_TEST = []


@pytest.mark.parametrize("estimator", EST_TO_TEST)
def test_sktime_estimator(estimator):
    """Test sktime estimator via check_estimator."""
    from sktime.utils.estimator_checks import check_estimator

    check_estimator(estimator, raise_exceptions=True)
    # The above line collects all API conformance tests in sktime and runs them.
    # It will raise an error if the estimator is not API conformant.


@pytest.mark.skipif(
    not _check_soft_dependencies("sktime", severity="none"),
    reason="sktime not installed",
)
def test_forecasting_opt_cv_sets_attributes():
    """ForecastingOptCV exposes useful attributes after fitting."""
    fh = [1, 2]
    y = load_airline().iloc[:36]
    cv = ExpandingWindowSplitter(initial_window=24, step_length=6, fh=fh)
    optimizer = GridSearchSk(param_grid={"strategy": ["last", "mean"]})

    tuner = ForecastingOptCV(
        forecaster=NaiveForecaster(),
        optimizer=optimizer,
        cv=cv,
        scoring=MeanAbsolutePercentageError(symmetric=True),
        backend="None",
    )

    tuner.fit(y=y, fh=fh)

    assert tuner.scorer_.name == "MeanAbsolutePercentageError"
    assert tuner.n_splits_ == cv.get_n_splits(y)
    assert tuner.refit_time_ >= 0

    metric_col = "test_" + tuner.scorer_.name
    assert metric_col in tuner.cv_results_.columns
    assert np.isclose(tuner.best_score_, tuner.cv_results_[metric_col].mean())


@pytest.mark.skipif(
    not _check_soft_dependencies("sktime", severity="none"),
    reason="sktime not installed",
)
def test_forecasting_opt_cv_tune_by_flags():
    """Tune-by flags should adjust estimator tags."""
    tuner = ForecastingOptCV(
        forecaster=NaiveForecaster(),
        optimizer=GridSearchSk(param_grid={"strategy": ["last"]}),
        cv=ExpandingWindowSplitter(initial_window=5, step_length=1, fh=[1]),
        tune_by_instance=True,
        tune_by_variable=True,
    )

    assert tuner.get_tag("scitype:y") == "univariate"
    y_mtypes = tuner.get_tag("y_inner_mtype")
    assert "pd-multiindex" not in y_mtypes
    assert "pd_multiindex_hier" not in y_mtypes
