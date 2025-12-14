.. _user_guide_experiments:

===========
Experiments
===========

An experiment is the objective function that evaluates parameters and returns a score.
It encapsulates your optimization problem separately from the optimizer, so you can
swap algorithms without changing your evaluation code. Hyperactive supports custom
functions and built-in classes for common ML tasks with cross-validation.

----

What is an Experiment?
----------------------

An experiment is any function that takes parameters and returns a score.
Hyperactive will search for parameters that **maximize** this score.

.. raw:: html

   <div class="theme-aware-diagram">
      <img src="../_static/diagrams/experiment_flow_light.svg"
           alt="Experiment flow: parameters go in, score comes out"
           class="only-light" />
      <img src="../_static/diagrams/experiment_flow_dark.svg"
           alt="Experiment flow: parameters go in, score comes out"
           class="only-dark" />
   </div>

|

.. code-block:: python

   def experiment(params):
       # Your evaluation logic here
       return score  # Hyperactive maximizes this

----

Two Approaches
--------------

Choose based on your use case:

.. grid:: 1 2 2 2
   :gutter: 4

   .. grid-item-card:: Custom Functions
      :class-card: sd-border-primary

      **For any optimization problem**
      ^^^
      Write a function that evaluates parameters and returns a score.
      Full flexibility for simulations, engineering, research, or any custom logic.

      .. code-block:: python

         def experiment(params):
             result = run_simulation(params)
             return result.quality

   .. grid-item-card:: Built-in Classes
      :class-card: sd-border-success

      **For machine learning tasks**
      ^^^
      Pre-built experiments for sklearn, sktime, and PyTorch.
      Handles cross-validation, scoring, and best practices automatically.

      .. code-block:: python

         from hyperactive.experiment.integrations import SklearnCvExperiment

         experiment = SklearnCvExperiment(model, X, y, cv=5)

----

Custom Functions
----------------

The simplest form of experiment. Takes a dictionary of parameters and returns a number.

Basic Example
^^^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:simple_objective]
   :end-before: # [end:simple_objective]

.. important::

   **Hyperactive maximizes the score.** To minimize a loss, negate it:

   .. code-block:: python

      return -loss  # Convert minimization to maximization

Mathematical Functions
^^^^^^^^^^^^^^^^^^^^^^

Optimizing benchmark or mathematical functions:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:ackley_function]
   :end-before: # [end:ackley_function]

External Simulations
^^^^^^^^^^^^^^^^^^^^

Your function can call any Python code, including simulations, APIs, or file I/O:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:external_simulation]
   :end-before: # [end:external_simulation]

----

Built-in Experiments
--------------------

For common ML tasks, Hyperactive provides ready-to-use experiment classes.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Class
     - Use Case
     - Install
   * - ``SklearnCvExperiment``
     - Scikit-learn models with CV
     - Included
   * - ``SktimeForecastingExperiment``
     - Time series forecasting
     - ``pip install hyperactive[sktime]``
   * - ``TorchTrainerExperiment``
     - PyTorch Lightning models
     - ``pip install hyperactive[torch]``

SklearnCvExperiment
^^^^^^^^^^^^^^^^^^^

The most common choice for tuning sklearn classifiers and regressors.

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:sklearn_cv_experiment]
   :end-before: # [end:sklearn_cv_experiment]

**Key parameters:**

- ``estimator``: Any sklearn estimator
- ``X, y``: Your training data
- ``cv``: Number of cross-validation folds (default: 5)
- ``scoring``: Scoring metric (default: estimator's default)

SktimeForecastingExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For time series forecasting optimization:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:sktime_forecasting]
   :end-before: # [end:sktime_forecasting]

TorchTrainerExperiment
^^^^^^^^^^^^^^^^^^^^^^

For PyTorch Lightning model optimization:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:torch_experiment]
   :end-before: # [end:torch_experiment]

----

Benchmark Functions
-------------------

Standard test functions for evaluating optimizers:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:benchmark_experiments]
   :end-before: # [end:benchmark_experiments]

Available benchmarks include: Ackley, Rastrigin, Rosenbrock, Sphere, and more.
These are useful for comparing optimizer performance on known landscapes.

----

Direct Evaluation
-----------------

Experiments can be evaluated directly using the ``score()`` method:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:score_method]
   :end-before: # [end:score_method]

This is useful for debugging or manual exploration before running optimization.

----

Error Handling
--------------

Robust experiments handle invalid parameter combinations gracefully:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:robust_objective]
   :end-before: # [end:robust_objective]

.. tip::

   Returning ``-np.inf`` signals an invalid configuration.
   The optimizer will learn to avoid this region of the search space.

----

Best Practices
--------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Return Meaningful Scores

      Ensure your score reflects what you want to optimize.
      Higher is better (Hyperactive maximizes).

   .. grid-item-card:: Handle Errors Gracefully

      Return ``-np.inf`` for invalid configurations instead
      of raising exceptions.

   .. grid-item-card:: Consider Compute Time

      For expensive experiments, use ``BayesianOptimizer`` or
      ``TPEOptimizer`` which learn from previous evaluations.

   .. grid-item-card:: Use Reproducibility

      Set random seeds inside your experiment for consistent
      results across runs.

----

Quick Reference
---------------

.. code-block:: python

   # Custom function
   def experiment(params):
       result = evaluate(params)
       return score  # Higher is better

   # Sklearn integration
   from hyperactive.experiment.integrations import SklearnCvExperiment
   experiment = SklearnCvExperiment(model, X, y, cv=5)

   # Use with any optimizer
   from hyperactive.opt.gfo import HillClimbing
   optimizer = HillClimbing(search_space, experiment=experiment)
   best = optimizer.solve()
