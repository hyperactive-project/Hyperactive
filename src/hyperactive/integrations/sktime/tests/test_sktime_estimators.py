"""Integration tests for sktime tuners."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("sktime", severity="none"):
    from hyperactive.integrations.sktime import ForecastingOptCV, TSCOptCV

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
