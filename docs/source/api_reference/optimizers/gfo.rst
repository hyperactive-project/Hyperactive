.. _optimizers_gfo_ref:

Gradient-Free Optimizers
========================

.. currentmodule:: hyperactive.opt

The GFO backend provides optimization algorithms from the
`gradient-free-optimizers <https://github.com/SimonBlanworthe/Gradient-Free-Optimizers>`_ package.
These implement various metaheuristic and numerical optimization algorithms.

Local Search
------------

Local search algorithms explore the neighborhood of the current position
to find improving solutions.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    HillClimbing
    StochasticHillClimbing
    RepulsingHillClimbing
    RandomRestartHillClimbing

Simulated Annealing
-------------------

Probabilistic technique for approximating the global optimum by allowing
occasional moves to worse solutions to escape local optima.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    SimulatedAnnealing

Global Search
-------------

Random and systematic search strategies that explore the search space broadly.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    RandomSearch
    GridSearch

Direct Methods
--------------

Direct search methods that do not require gradient information and work
directly with function evaluations.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    DownhillSimplexOptimizer
    PowellsMethod
    PatternSearch
    LipschitzOptimizer
    DirectAlgorithm

Population-Based
----------------

Optimization algorithms that maintain and evolve populations of candidate solutions.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    ParallelTempering
    ParticleSwarmOptimizer
    SpiralOptimization
    GeneticAlgorithm
    EvolutionStrategy
    DifferentialEvolution

Sequential Model-Based
----------------------

Algorithms that build surrogate models of the objective function to guide
the search towards promising regions.

.. autosummary::
    :toctree: ../auto_generated/
    :template: class.rst

    BayesianOptimizer
    TreeStructuredParzenEstimators
    ForestOptimizer
