"""Integrations with packages for tuning."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.experiment.integrations.sklearn_cv import SklearnCvExperiment
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
from hyperactive.experiment.integrations.sktime_detector import (
    SktimeDetectorExperiment,
)

__all__ = [
    "SklearnCvExperiment",
    "SkproProbaRegExperiment",
    "SktimeClassificationExperiment",
    "SktimeForecastingExperiment",
    "SktimeDetectorExperiment",
    "TorchExperiment",
]
