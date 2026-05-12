.. _optimizers_smac_ref:

SMAC
====

.. currentmodule:: hyperactive.opt

The SMAC backend provides an interface to `SMAC3 <https://automl.github.io/SMAC3/main/>`_
(Sequential Model-based Algorithm Configuration) optimization algorithms.
These optimizers use Bayesian optimization with different surrogate models.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    SmacRandomForest
    SmacGaussianProcess
    SmacRandomSearch
