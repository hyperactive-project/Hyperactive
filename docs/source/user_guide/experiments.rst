.. _user_guide_experiments:

===========
Experiments
===========

Experiments define *what* to optimize in Hyperactive. They encapsulate the objective
function and any evaluation logic needed to score a set of parameters.

Defining Experiments
--------------------

There are two ways to define experiments in Hyperactive:

1. **Custom functions** — Simple callables for any optimization problem
2. **Built-in experiment classes** — Pre-built experiments for common ML tasks


Custom Objective Functions
--------------------------

The simplest way to define an experiment is as a Python function that takes
a parameter dictionary and returns a score:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:simple_objective]
   :end-before: # [end:simple_objective]

Key points:

- The function receives a dictionary with parameter names as keys
- It must return a single numeric value (the score)
- Hyperactive **maximizes** this score by default
- To minimize, negate your loss function (as shown above)


Example: Optimizing a Mathematical Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:ackley_function]
   :end-before: # [end:ackley_function]


Example: Optimizing with External Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your objective function can use any Python code:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:external_simulation]
   :end-before: # [end:external_simulation]


Built-in Experiment Classes
---------------------------

For common machine learning tasks, Hyperactive provides ready-to-use experiment classes
that handle cross-validation, scoring, and other details.


SklearnCvExperiment
^^^^^^^^^^^^^^^^^^^

For optimizing scikit-learn estimators with cross-validation:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:sklearn_cv_experiment]
   :end-before: # [end:sklearn_cv_experiment]


SktimeForecastingExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For time series forecasting optimization (requires ``sktime``):

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:sktime_forecasting]
   :end-before: # [end:sktime_forecasting]


TorchExperiment
^^^^^^^^^^^^^^^

For PyTorch Lightning model optimization (requires ``lightning``):

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:torch_experiment]
   :end-before: # [end:torch_experiment]


Benchmark Experiments
---------------------

Hyperactive includes standard benchmark functions for testing optimizers:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:benchmark_experiments]
   :end-before: # [end:benchmark_experiments]


Using the score() Method
------------------------

Experiments can also be evaluated directly using the ``score()`` method:

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:score_method]
   :end-before: # [end:score_method]


Tips for Designing Experiments
------------------------------

1. **Return meaningful scores**: Ensure your score reflects what you want to optimize.
   Higher is better (Hyperactive maximizes).

2. **Handle errors gracefully**: If a parameter combination fails, return a very
   low score (e.g., ``-np.inf``) rather than raising an exception.

3. **Consider computation time**: For expensive experiments, use efficient optimizers
   like ``BayesianOptimizer`` that learn from previous evaluations.

4. **Use reproducibility**: Set random seeds in your experiment for consistent results.

.. literalinclude:: ../_snippets/user_guide/experiments.py
   :language: python
   :start-after: # [start:robust_objective]
   :end-before: # [end:robust_objective]
