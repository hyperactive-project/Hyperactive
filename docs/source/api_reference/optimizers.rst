.. _optimizers_ref:

Optimizers
==========

The :mod:`hyperactive.opt` module contains optimization algorithms for hyperparameter tuning.

All optimizers inherit from :class:`hyperactive.base.BaseOptimizer` and share the same interface:
the ``solve()`` method to run optimization, and configuration via the ``experiment`` and ``search_space`` parameters.

Gradient-Free Optimizers (GFO)
-------------------------------

These optimizers are based on the gradient-free-optimizers package and implement
various metaheuristic and numerical optimization algorithms.

Local Search
~~~~~~~~~~~~

Local search algorithms that explore the neighborhood of the current position.

.. currentmodule:: hyperactive.opt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HillClimbing
    StochasticHillClimbing
    RepulsingHillClimbing
    RandomRestartHillClimbing

Simulated Annealing
~~~~~~~~~~~~~~~~~~~

Probabilistic techniques for approximating the global optimum.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SimulatedAnnealing

Global Search
~~~~~~~~~~~~~

Random and systematic search strategies.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomSearch
    GridSearch

Direct Methods
~~~~~~~~~~~~~~

Direct search methods that do not use gradients.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DownhillSimplexOptimizer
    PowellsMethod
    PatternSearch
    LipschitzOptimizer
    DirectAlgorithm

Population-Based
~~~~~~~~~~~~~~~~

Optimization algorithms that maintain and evolve populations of solutions.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ParallelTempering
    ParticleSwarmOptimizer
    SpiralOptimization
    GeneticAlgorithm
    EvolutionStrategy
    DifferentialEvolution

Sequential Model-Based
~~~~~~~~~~~~~~~~~~~~~~

Algorithms that build surrogate models of the objective function.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BayesianOptimizer
    TreeStructuredParzenEstimators
    ForestOptimizer

Optuna-Based Optimizers
------------------------

These optimizers provide an interface to Optuna's optimization algorithms.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TPEOptimizer
    RandomOptimizer
    CmaEsOptimizer
    GPOptimizer
    GridOptimizer
    NSGAIIOptimizer
    NSGAIIIOptimizer
    QMCOptimizer

Scikit-Learn Style
-------------------

Optimizers with a scikit-learn compatible interface.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GridSearchSk
    RandomSearchSk
