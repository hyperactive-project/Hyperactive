.. _user_guide_introduction:

============
Introduction
============

Optimization finds the best parameters from a set of possibilities. Evaluating every
combination is usually impractical, so algorithms guide the search. Hyperactive provides
a unified interface: define your problem once, then swap between different optimization
algorithms without changing your code.

----

Why Hyperactive?
----------------

Hyperactive makes optimization simple. Define your problem once, then swap between
31 different algorithms with a single line change.

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: Swap Algorithms
      :class-card: sd-bg-light

      Try different optimizers without rewriting code. One line change switches
      from hill climbing to Bayesian optimization.

   .. grid-item-card:: 31 Algorithms
      :class-card: sd-bg-light

      Local search, global search, population-based, and model-based methods.
      All with the same simple interface.

   .. grid-item-card:: ML Integrations
      :class-card: sd-bg-light

      Ready-to-use experiments for sklearn, sktime, skpro, and PyTorch.
      Tune models with minimal code.

   .. grid-item-card:: Any Python Function
      :class-card: sd-bg-light

      Works with simulations, engineering problems, or any function that
      returns a score.

----

Quick Start
-----------

The simplest optimization in 5 lines:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:simplest_example]
   :end-before: # [end:simplest_example]

----

ML Integration Examples
-----------------------

Tune machine learning models with ready-to-use experiment classes:

.. tab-set::

   .. tab-item:: sklearn

      .. code-block:: python

         from hyperactive.experiment.integrations import SklearnCvExperiment
         from sklearn.ensemble import GradientBoostingClassifier

         experiment = SklearnCvExperiment(GradientBoostingClassifier(), X, y, cv=5)

   .. tab-item:: sktime

      .. code-block:: python

         from hyperactive.experiment.integrations import SktimeForecastingExperiment
         from sktime.forecasting.arima import ARIMA

         experiment = SktimeForecastingExperiment(ARIMA(), y_train, fh=[1, 2, 3])

   .. tab-item:: PyTorch

      .. code-block:: python

         from hyperactive.experiment.integrations import TorchTrainerExperiment
         import pytorch_lightning as pl

         experiment = TorchTrainerExperiment(YourLightningModule, train_loader)

See :doc:`integrations` for complete examples and all available integrations.

----

Core Concepts
-------------

Hyperactive is built around three simple concepts:

.. raw:: html

   <div class="theme-aware-diagram">
      <img src="../_static/diagrams/intro_concepts_light.svg"
           alt="Hyperactive core concepts: Experiment, Search Space, and Optimizer flow to Best Parameters"
           class="only-light" />
      <img src="../_static/diagrams/intro_concepts_dark.svg"
           alt="Hyperactive core concepts: Experiment, Search Space, and Optimizer flow to Best Parameters"
           class="only-dark" />
   </div>

|

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: Experiment
      :class-card: sd-border-primary

      **What to optimize**
      ^^^
      Any Python function that takes parameters and returns a score.
      Hyperactive will maximize this score.

      .. code-block:: python

         def experiment(params):
             return score

      :doc:`Learn more <experiments>`

   .. grid-item-card:: Search Space
      :class-card: sd-border-success

      **Where to search**
      ^^^
      A dictionary mapping parameter names to possible values.
      Defines the boundaries of your optimization.

      .. code-block:: python

         {"x": [1, 2, 3], "y": ["a", "b"]}

      :doc:`Learn more <search_spaces>`

   .. grid-item-card:: Optimizer
      :class-card: sd-border-warning

      **How to search**
      ^^^
      The algorithm that explores the search space.
      Choose based on your problem characteristics.

      .. code-block:: python

         HillClimbing(space, experiment=exp)

      :doc:`Learn more <optimizers/index>`

----

The Power of Swapping
---------------------

Define your problem once, then try different algorithms with a single line change:

.. tab-set::

   .. tab-item:: Hill Climbing

      .. literalinclude:: ../_snippets/user_guide/introduction.py
         :language: python
         :start-after: # [start:swap_hill_climbing]
         :end-before: # [end:swap_hill_climbing]

      Fast local search. Good for quick exploration.

   .. tab-item:: Bayesian Optimization

      .. literalinclude:: ../_snippets/user_guide/introduction.py
         :language: python
         :start-after: # [start:swap_bayesian]
         :end-before: # [end:swap_bayesian]

      Learns from past evaluations. Best for expensive functions.

   .. tab-item:: Genetic Algorithm

      .. literalinclude:: ../_snippets/user_guide/introduction.py
         :language: python
         :start-after: # [start:swap_genetic]
         :end-before: # [end:swap_genetic]

      Population-based evolution. Great for complex landscapes.

.. tip::

   The experiment and search space stay the same. Only the optimizer changes.
   This makes it easy to benchmark different algorithms on your problem.

----

Complete Example
----------------

Here's a full working example that tunes a Random Forest classifier:

.. literalinclude:: ../_snippets/user_guide/introduction.py
   :language: python
   :start-after: # [start:complete_example]
   :end-before: # [end:complete_example]

----

Common Parameters
-----------------

All optimizers share these parameters:

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

----

Tips for Beginners
------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Start Simple

      Begin with ``HillClimbing`` or ``RandomSearch`` to establish baselines
      before trying sophisticated algorithms.

   .. grid-item-card:: Right-Size Your Space

      Large search spaces need more iterations. Use ``np.logspace`` for
      parameters that span orders of magnitude.

   .. grid-item-card:: Set Random State

      For reproducible results, always set ``random_state=42`` (or any integer).

   .. grid-item-card:: Match Algorithm to Budget

      Expensive evaluations? Use ``BayesianOptimizer`` which learns from
      each evaluation. Cheap evaluations? ``RandomSearch`` explores well.

----

Next Steps
----------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Experiments
      :link: experiments
      :link-type: doc

      Learn how to define what to optimize, including custom functions
      and built-in ML experiments.

   .. grid-item-card:: Search Spaces
      :link: search_spaces
      :link-type: doc

      Master parameter definitions, scaling strategies, and space sizing.

   .. grid-item-card:: Optimizers
      :link: optimizers/index
      :link-type: doc

      Explore all 31 algorithms and learn when to use each one.
