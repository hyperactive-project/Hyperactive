.. _user_guide_introduction:

============
Introduction
============

This page introduces Hyperactive's core concepts: optimizers, experiments, and search spaces.
Understanding these concepts will help you use Hyperactive effectively for any optimization task.


Core Concepts
-------------

Hyperactive is built around three key concepts:

1. **Experiments** — Define *what* to optimize (the objective function)
2. **Optimizers** — Define *how* to optimize (the search algorithm)
3. **Search Spaces** — Define *where* to search (the parameter ranges)


Experiments: What to Optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **experiment** represents your optimization problem. It takes parameters as input
and returns a score that Hyperactive will maximize.

The simplest experiment is a Python function:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:simple_objective]
   :end-before: # [end:simple_objective]

For machine learning, Hyperactive provides built-in experiments:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:sklearn_experiment_intro]
   :end-before: # [end:sklearn_experiment_intro]

See :ref:`user_guide_experiments` for more details.


Optimizers: How to Optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **optimizer** is the algorithm that explores the search space to find the best parameters.
Hyperactive provides 20+ optimizers in different categories:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:optimizer_imports]
   :end-before: # [end:optimizer_imports]

Each optimizer has different characteristics:

- **Local search** (HillClimbing, SimulatedAnnealing): Fast, may get stuck in local optima
- **Global search** (RandomSearch, GridSearch): Thorough exploration, slower
- **Population methods** (GeneticAlgorithm, ParticleSwarm): Good for complex landscapes
- **Sequential methods** (BayesianOptimizer, TPE): Smart exploration, best for expensive evaluations

See :ref:`user_guide_optimizers` for a complete guide.


Search Spaces: Where to Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **search space** defines the possible values for each parameter.
Use dictionaries with lists or NumPy arrays:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:search_space_definition]
   :end-before: # [end:search_space_definition]

.. tip::

    Keep search spaces reasonably sized. Very large spaces (>10^8 combinations)
    can cause memory issues with some optimizers.


Basic Workflow
--------------

Here's the complete workflow for using Hyperactive:

Step 1: Define Your Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Either as a function or using built-in experiment classes:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:workflow_experiment_options]
   :end-before: # [end:workflow_experiment_options]


Step 2: Define the Search Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:workflow_search_space]
   :end-before: # [end:workflow_search_space]


Step 3: Choose an Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:workflow_optimizer]
   :end-before: # [end:workflow_optimizer]


Step 4: Run the Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:workflow_solve]
   :end-before: # [end:workflow_solve]


Common Optimizer Parameters
---------------------------

Most optimizers share these parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``search_space``
     - dict
     - Maps parameter names to possible values
   * - ``n_iter``
     - int
     - Number of optimization iterations
   * - ``experiment``
     - callable
     - The objective function or experiment object
   * - ``random_state``
     - int
     - Seed for reproducibility
   * - ``initialize``
     - dict
     - Control initial population (warm starts, etc.)


Warm Starting
^^^^^^^^^^^^^

You can provide starting points for optimization:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:warm_starting]
   :end-before: # [end:warm_starting]


Tips for Effective Optimization
-------------------------------

1. **Start simple**: Begin with ``HillClimbing`` or ``RandomSearch`` to establish baselines.

2. **Right-size your search space**: Large spaces need more iterations. Consider using
   ``np.logspace`` for parameters that span orders of magnitude.

3. **Use appropriate iterations**: More iterations = better exploration, but longer runtime.
   A good rule of thumb: at least 10x the number of parameters.

4. **Set random_state**: For reproducible results, always set a random seed.

5. **Consider your budget**: For expensive evaluations (training large models),
   use smart optimizers like ``BayesianOptimizer`` that learn from previous evaluations.
