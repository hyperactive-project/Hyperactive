"""Integrations with packages for tuning."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.integrations.sklearn_cv import (
    SklearnCvExperiment,
    XGBoostCvExperiment,
)
from hyperactive.experiment.integrations.skpro_probareg import (
    SkproProbaRegExperiment,
)
from hyperactive.experiment.integrations.sktime_classification import (
    SktimeClassificationExperiment,
)
from hyperactive.experiment.integrations.sktime_forecasting import (
    SktimeForecastingExperiment,
)
from hyperactive.experiment.integrations.torch_lightning_experiment import (
    TorchExperiment,
)

__all__ = [
    "SklearnCvExperiment",
    "SkproProbaRegExperiment",
    "SktimeClassificationExperiment",
    "SktimeForecastingExperiment",
    "TorchExperiment",
    "XGBoostCvExperiment",
]
