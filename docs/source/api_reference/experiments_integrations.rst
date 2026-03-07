.. _experiments_integrations_ref:

Framework Integration Experiments
==================================

The :mod:`hyperactive.experiment.integrations` module contains experiment classes
for integration with machine learning frameworks.

These experiments provide seamless hyperparameter optimization for scikit-learn,
sktime, skpro, and PyTorch Lightning models.

Scikit-Learn
------------

Cross-validation experiments for scikit-learn estimators.

.. currentmodule:: hyperactive.experiment.integrations

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SklearnCvExperiment

Sktime - Time Series
--------------------

Experiments for sktime time series models.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SktimeClassificationExperiment
    SktimeForecastingExperiment

Skpro - Probabilistic Prediction
---------------------------------

Experiments for skpro probabilistic regression models.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SkproProbaRegExperiment

PyTorch Lightning
-----------------

Experiments for PyTorch Lightning models.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TorchExperiment
