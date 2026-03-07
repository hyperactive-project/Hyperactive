.. _get_started:

===========
Get Started
===========

This guide will help you get up and running with Hyperactive in just a few minutes.
By the end, you'll understand the core concepts and be able to run your first optimization.

Quick Start
-----------

Hyperactive makes hyperparameter optimization simple. Here's a complete example
that optimizes a custom function:

.. literalinclude:: _snippets/getting_started/quick_start.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]

That's it! Let's break down what happened:

1. **Objective function**: A callable that takes a dictionary of parameters and returns a score.
   Hyperactive **maximizes** this score by default.

2. **Search space**: A dictionary mapping parameter names to their possible values.
   Use NumPy arrays or lists to define discrete search spaces.

3. **Optimizer**: Choose from 20+ optimization algorithms. Each optimizer explores the
   search space differently to find optimal parameters.


First Steps
-----------

Optimizing a Scikit-learn Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common use case is tuning machine learning models. Here's how to optimize
a Random Forest classifier:

.. literalinclude:: _snippets/getting_started/sklearn_random_forest.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]


Using the Sklearn Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For even simpler sklearn integration, use the ``OptCV`` wrapper that behaves like
scikit-learn's ``GridSearchCV``:

.. literalinclude:: _snippets/getting_started/sklearn_optcv.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]


Choosing an Optimizer
^^^^^^^^^^^^^^^^^^^^^

Hyperactive provides many optimization algorithms. Here are some common choices:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Optimizer
     - Best For
   * - ``HillClimbing``
     - Quick local optimization, good starting point
   * - ``RandomSearch``
     - Exploring large search spaces, baseline comparison
   * - ``BayesianOptimizer``
     - Expensive evaluations, smart exploration
   * - ``ParticleSwarmOptimizer``
     - Multi-modal problems, avoiding local optima
   * - ``GeneticAlgorithm``
     - Complex landscapes, combinatorial problems

Example with Bayesian Optimization:

.. literalinclude:: _snippets/getting_started/bayesian_optimizer.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]

.. literalinclude:: _snippets/getting_started/bayesian_optimizer.py
   :language: python
   :start-after: # [start:optimizer_usage]
   :end-before: # [end:optimizer_usage]


Next Steps
----------

Now that you've seen the basics, explore these topics:

- :ref:`installation` - Detailed installation instructions
- :ref:`user_guide` - In-depth tutorials and concepts
- :ref:`api_reference` - Complete API documentation
- :ref:`examples` - More code examples

Key Concepts to Learn
^^^^^^^^^^^^^^^^^^^^^

1. **Experiments**: Abstractions that define *what* to optimize (see :ref:`user_guide_experiments`)
2. **Optimizers**: Algorithms that define *how* to optimize (see :ref:`user_guide_optimizers`)
3. **Search Spaces**: Define the parameter ranges to explore
4. **Integrations**: Built-in support for sklearn, sktime, and PyTorch (see :ref:`user_guide_integrations`)
