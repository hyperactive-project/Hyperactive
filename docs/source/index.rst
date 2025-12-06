.. _home:

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <h1 class="hero-title">Hyperactive</h1>
         <p class="hero-tagline">A unified interface for optimization algorithms and experiments</p>
      </div>
   </div>

.. raw:: html

   <div class="badge-banner">
      <div class="badge-items">
         <a href="https://pypi.org/project/hyperactive/" target="_blank">
            <img src="_static/images/badges/version.svg" alt="PyPI Version" />
         </a>
         <a href="https://pypi.org/project/hyperactive/" target="_blank">
            <img src="_static/images/badges/python.svg" alt="Python Versions" />
         </a>
         <a href="https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE" target="_blank">
            <img src="_static/images/badges/license.svg" alt="License" />
         </a>
         <a href="https://gc-os-ai.github.io/" target="_blank">
            <img src="_static/images/badges/sponsor.svg" alt="GC.OS Sponsored" />
         </a>
      </div>
   </div>


Hyperactive provides a collection of optimization algorithms, accessible through a unified
experiment-based interface that separates optimization problems from algorithms. The library
provides native implementations of algorithms from the Gradient-Free-Optimizers package
alongside direct interfaces to Optuna and scikit-learn optimizers.

----

Why Hyperactive?
================

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card::
      :class-card: feature-card

      **20 Optimization Algorithms**
      ^^^
      From Hill Climbing to Bayesian Optimization,
      Particle Swarm, Genetic Algorithms, and more.

      +++
      :doc:`Local, global, population-based, and sequential methods <api_reference/optimizers>`

   .. grid-item-card::
      :class-card: feature-card

      **Direct ML Integration**
      ^^^
      First-class support for scikit-learn, sktime, skpro, and PyTorch.
      Tune models with minimal code changes.

      +++
      :doc:`Works with any estimator implementing fit/score <user_guide/integrations>`

   .. grid-item-card::
      :class-card: feature-card

      **Experiment Abstraction**
      ^^^
      Clean separation between *what* to optimize (experiments) and
      *how* to optimize (algorithms).

      +++
      :doc:`Swap algorithms without changing experiment code <user_guide/experiments>`

   .. grid-item-card::
      :class-card: feature-card

      **3 Optimizer Backends**
      ^^^
      Native GFO algorithms, Optuna samplers, and scikit-learn
      search methods through one unified API.

      +++
      :doc:`GFO · Optuna · sklearn <user_guide/optimizers>`

   .. grid-item-card::
      :class-card: feature-card

      **Mixed Parameter Spaces**
      ^^^
      Categorical, integer, and continuous parameters.
      Define search spaces with NumPy arrays or lists.

      +++
      :doc:`Optuna backend supports native continuous ranges <user_guide/search_spaces>`

   .. grid-item-card::
      :class-card: feature-card

      **Stable Since 2019**
      ^^^
      5+ years of development, comprehensive test coverage,
      and active maintenance.

      +++
      :doc:`Type-annotated · Documented · Tested <about/history>`

----

Quick Example
=============

Get started in just a few lines of code:

.. tab-set::

   .. tab-item:: Custom Function

      .. literalinclude:: _snippets/getting_started/index_custom_function.py
         :language: python
         :start-after: # [start:full_example]
         :end-before: # [end:full_example]

   .. tab-item:: Scikit-learn Tuning

      .. literalinclude:: _snippets/getting_started/index_sklearn_tuning.py
         :language: python
         :start-after: # [start:full_example]
         :end-before: # [end:full_example]

   .. tab-item:: Bayesian Optimization

      .. literalinclude:: _snippets/getting_started/index_bayesian.py
         :language: python
         :start-after: # [start:full_example]
         :end-before: # [end:full_example]

----

.. raw:: html

   <div class="visualization-section">
      <h2>Visualization</h2>
      <p>Watch Bayesian Optimization intelligently explore parameter space:</p>
      <img src="_static/images/bayes_convex.gif" alt="Bayesian Optimization Animation" class="optimization-gif" />
   </div>

----

Optimizer Backends
==================

Hyperactive provides a unified interface to three powerful optimization backends.
Choose the one that best fits your needs, or switch between them effortlessly.

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card::
      :class-card: backend-card backend-card-gfo

      `Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`__
      ^^^
      The native backend with 20 optimization algorithms implemented from scratch.
      Ideal for custom objective functions and research applications.

      - Hill Climbing variants
      - Simulated Annealing
      - Particle Swarm & Genetic Algorithms
      - Bayesian Optimization
      - And 15+ more algorithms

      +++
      :doc:`Explore GFO algorithms <examples/general>`

   .. grid-item-card::
      :class-card: backend-card backend-card-optuna

      `Optuna <https://github.com/optuna/optuna>`__
      ^^^
      Industry-standard hyperparameter optimization framework with
      state-of-the-art samplers and pruning strategies.

      - Tree-Parzen Estimator (TPE)
      - CMA-ES for continuous spaces
      - Gaussian Process optimization
      - Multi-objective (NSGA-II/III)
      - Native continuous parameter support

      +++
      :doc:`Explore Optuna samplers <examples/optuna_backend>`

   .. grid-item-card::
      :class-card: backend-card backend-card-sklearn

      `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__
      ^^^
      Familiar scikit-learn search interfaces with enhanced integration
      for cross-validation experiments.

      - GridSearchCV
      - RandomizedSearchCV
      - HalvingGridSearchCV
      - HalvingRandomSearchCV
      - Drop-in sklearn compatibility

      +++
      :doc:`Explore sklearn integration <examples/sklearn_backend>`

----

Integrations
============

.. raw:: html

   <div class="integration-grid">
      <div class="integration-card sklearn">
         <div class="integration-left">
            <a href="https://github.com/scikit-learn/scikit-learn" target="_blank">scikit-learn</a>
         </div>
         <div class="integration-right">
            <a href="user_guide/integrations.html#scikit-learn-integration">
               <span class="integration-desc">Tune estimators with OptCV cross-validation</span>
            </a>
         </div>
      </div>
      <div class="integration-card sktime">
         <div class="integration-left">
            <a href="https://github.com/sktime/sktime" target="_blank">sktime</a>
         </div>
         <div class="integration-right">
            <a href="user_guide/integrations.html#time-series-with-sktime">
               <span class="integration-desc">Optimize forecasters and transformers</span>
            </a>
         </div>
      </div>
      <div class="integration-card skpro">
         <div class="integration-left">
            <a href="https://github.com/sktime/skpro" target="_blank">skpro</a>
         </div>
         <div class="integration-right">
            <a href="user_guide/integrations.html#probabilistic-models-with-skpro">
               <span class="integration-desc">Tune probabilistic regressors</span>
            </a>
         </div>
      </div>
      <div class="integration-card pytorch">
         <div class="integration-left">
            <a href="https://github.com/pytorch/pytorch" target="_blank">PyTorch</a>
         </div>
         <div class="integration-right">
            <a href="user_guide/integrations.html#deep-learning-with-pytorch">
               <span class="integration-desc">Optimize architectures and training parameters</span>
            </a>
         </div>
      </div>
   </div>

----

Quick Install
=============

.. code-block:: bash

   pip install hyperactive

For additional integrations:

.. code-block:: bash

   # Full installation with all extras
   pip install hyperactive[all_extras]

   # Or specific integrations
   pip install hyperactive[sklearn-integration]
   pip install hyperactive[sktime-integration]

----

Contents
========

.. toctree::
   :maxdepth: 1
   :hidden:

   get_started
   installation
   user_guide
   api_reference
   examples
   faq
   troubleshooting
   get_involved
   about

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`rocket` **Get Started**
      ^^^

      Quick introduction to using Hyperactive.

      +++

      .. button-ref:: get_started
         :color: primary
         :click-parent:
         :expand:

         Get Started

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`download` **Installation**
      ^^^

      Installation guide and requirements.

      +++

      .. button-ref:: installation
         :color: primary
         :click-parent:
         :expand:

         Installation

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`book` **User Guide**
      ^^^

      In-depth tutorials and explanations.

      +++

      .. button-ref:: user_guide
         :color: primary
         :click-parent:
         :expand:

         User Guide

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`code` **API Reference**
      ^^^

      Technical reference for all classes.

      +++

      .. button-ref:: api_reference
         :color: primary
         :click-parent:
         :expand:

         API Reference

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`laptop-code` **Examples**
      ^^^

      Code examples and use cases.

      +++

      .. button-ref:: examples
         :color: primary
         :click-parent:
         :expand:

         Examples

   .. grid-item-card::
      :text-align: center
      :class-card: nav-card

      :fas:`users` **Get Involved**
      ^^^

      Contribute to Hyperactive.

      +++

      .. button-ref:: get_involved
         :color: primary
         :click-parent:
         :expand:

         Get Involved
