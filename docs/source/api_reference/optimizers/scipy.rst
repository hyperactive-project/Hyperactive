.. _optimizers_scipy_ref:

Scipy
=====

.. currentmodule:: hyperactive.opt

The Scipy backend provides an interface to `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_
algorithms for continuous parameter optimization.

.. note::

    Scipy optimizers only support **continuous parameter spaces** (tuples).
    For discrete or categorical parameters, use GFO or Optuna backends.

Global Optimizers
-----------------

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    ScipyDifferentialEvolution
    ScipyDualAnnealing
    ScipyBasinhopping
    ScipySHGO
    ScipyDirect

Local Optimizers
----------------

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    ScipyNelderMead
    ScipyPowell
