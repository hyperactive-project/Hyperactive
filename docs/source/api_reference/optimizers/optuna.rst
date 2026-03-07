.. _optimizers_optuna_ref:

Optuna
======

.. currentmodule:: hyperactive.opt

The Optuna backend provides an interface to `Optuna's <https://optuna.org/>`_
optimization algorithms. These optimizers wrap Optuna's samplers and provide
access to state-of-the-art hyperparameter optimization methods.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    TPEOptimizer
    RandomOptimizer
    CmaEsOptimizer
    GPOptimizer
    GridOptimizer
    NSGAIIOptimizer
    NSGAIIIOptimizer
    QMCOptimizer
