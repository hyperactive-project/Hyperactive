.. _home:

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <h1 class="hero-title">Hyperactive</h1>
         <p class="hero-tagline">A unified interface for optimization algorithms and problems</p>
         <p class="hero-subtitle">Production-ready hyperparameter optimization for machine learning</p>
      </div>
   </div>

.. raw:: html

   <div class="maturity-banner">
      <div class="maturity-items">
         <div class="maturity-item">
            <span class="maturity-icon">🔧</span>
            <span class="maturity-text"><strong>Since 2019</strong></span>
         </div>
         <div class="maturity-item">
            <span class="maturity-icon">🚀</span>
            <span class="maturity-text"><strong>Version 5.0</strong></span>
         </div>
         <div class="maturity-item">
            <span class="maturity-icon">🐍</span>
            <span class="maturity-text"><strong>Python <span id="python-version-range"></span></strong></span>
         </div>
         <div class="maturity-item">
            <span class="maturity-icon">📜</span>
            <span class="maturity-text"><strong>MIT Licensed</strong></span>
         </div>
      </div>
   </div>
   <script>
      document.getElementById('python-version-range').textContent =
         document.querySelector('meta[name="python-version-range"]').content;
   </script>


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
      :text-align: center

      .. raw:: html

         <div class="feature-icon">⚡</div>

      **20+ Optimization Algorithms**
      ^^^
      From simple Hill Climbing to advanced Bayesian Optimization,
      Particle Swarm, Genetic Algorithms, and more. Choose the right
      algorithm for your problem.

   .. grid-item-card::
      :class-card: feature-card
      :text-align: center

      .. raw:: html

         <div class="feature-icon">🔌</div>

      **Seamless ML Integration**
      ^^^
      First-class support for scikit-learn, sktime, skpro, and PyTorch Lightning.
      Tune your models with minimal code changes.

   .. grid-item-card::
      :class-card: feature-card
      :text-align: center

      .. raw:: html

         <div class="feature-icon">🎯</div>

      **Experiment Abstraction**
      ^^^
      Clean separation between *what* to optimize (experiments) and
      *how* to optimize (algorithms). Mix and match freely.

   .. grid-item-card::
      :class-card: feature-card
      :text-align: center

      .. raw:: html

         <div class="feature-icon">🔧</div>

      **Multiple Backends**
      ^^^
      Native GFO implementations, Optuna integration, and scikit-learn
      optimizers — all through the same unified API.

   .. grid-item-card::
      :class-card: feature-card
      :text-align: center

      .. raw:: html

         <div class="feature-icon">📊</div>

      **Flexible Search Spaces**
      ^^^
      Discrete, continuous, and mixed parameter spaces. Define search
      spaces naturally using NumPy arrays or lists.

   .. grid-item-card::
      :class-card: feature-card
      :text-align: center

      .. raw:: html

         <div class="feature-icon">🏭</div>

      **Production Ready**
      ^^^
      Battle-tested since 2019 with comprehensive test coverage,
      active maintenance, and commercial sponsorship.

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

Available Algorithms
====================

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card::
      :class-card: algo-card

      **Local Search**
      ^^^
      - Hill Climbing
      - Repulsing Hill Climbing
      - Simulated Annealing
      - Downhill Simplex

   .. grid-item-card::
      :class-card: algo-card

      **Global Search**
      ^^^
      - Random Search
      - Grid Search
      - Random Restart Hill Climbing
      - Powell's Method
      - Pattern Search

   .. grid-item-card::
      :class-card: algo-card

      **Population Methods**
      ^^^
      - Parallel Tempering
      - Particle Swarm
      - Spiral Optimization
      - Genetic Algorithm
      - Evolution Strategy
      - Differential Evolution

   .. grid-item-card::
      :class-card: algo-card

      **Sequential / Bayesian**
      ^^^
      - Bayesian Optimization
      - Tree-Parzen Estimators
      - Forest Optimizer
      - Lipschitz Optimization
      - DIRECT Algorithm

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card::
      :class-card: algo-card optuna-card

      **Optuna Backend**
      ^^^
      - TPE Optimizer
      - CMA-ES
      - Gaussian Process
      - NSGA-II / NSGA-III
      - QMC Optimizer

----

Integrations
============

.. grid:: 1 2 4 4
   :gutter: 4

   .. grid-item::
      :class: integration-item

      .. raw:: html

         <div class="integration-card">
            <div class="integration-name">scikit-learn</div>
            <div class="integration-desc">Cross-validation experiments</div>
         </div>

   .. grid-item::
      :class: integration-item

      .. raw:: html

         <div class="integration-card">
            <div class="integration-name">sktime</div>
            <div class="integration-desc">Time series forecasting</div>
         </div>

   .. grid-item::
      :class: integration-item

      .. raw:: html

         <div class="integration-card">
            <div class="integration-name">skpro</div>
            <div class="integration-desc">Probabilistic regression</div>
         </div>

   .. grid-item::
      :class: integration-item

      .. raw:: html

         <div class="integration-card">
            <div class="integration-name">PyTorch</div>
            <div class="integration-desc">Deep learning models</div>
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

.. raw:: html

   <div class="sponsor-section">
      <a href="https://gc-os-ai.github.io/" target="_blank">
         <img src="https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=for-the-badge&colorA=0eac92&colorB=2077b4" alt="GC.OS Sponsored" />
      </a>
   </div>

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
