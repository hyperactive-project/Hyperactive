.. _home:

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <img src="_static/images/logo.png" alt="Hyperactive Logo" class="hero-logo" />
         <p class="hero-tagline">A unified interface for optimization algorithms and problems</p>
         <p class="hero-subtitle">Production-ready hyperparameter optimization for machine learning</p>
         <div class="hero-badges">
            <a href="https://pypi.org/project/hyperactive/"><img src="https://img.shields.io/pypi/v/hyperactive?style=for-the-badge&color=4c9aff" alt="PyPI Version" /></a>
            <a href="https://github.com/SimonBlanke/Hyperactive"><img src="https://img.shields.io/github/stars/SimonBlanke/Hyperactive?style=for-the-badge&color=ffd700" alt="GitHub Stars" /></a>
            <a href="https://github.com/SimonBlanke/Hyperactive/actions"><img src="https://img.shields.io/github/actions/workflow/status/SimonBlanke/hyperactive/test.yml?style=for-the-badge&logo=github" alt="Build Status" /></a>
         </div>
      </div>
   </div>

==========
Hyperactive
==========

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
            <span class="maturity-text"><strong>Python 3.10 - 3.14</strong></span>
         </div>
         <div class="maturity-item">
            <span class="maturity-icon">📜</span>
            <span class="maturity-text"><strong>MIT Licensed</strong></span>
         </div>
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

      .. code-block:: python

         import numpy as np
         from hyperactive.opt.gfo import HillClimbing

         # Define your objective function
         def objective(params):
             x, y = params["x"], params["y"]
             return -(x**2 + y**2)  # Maximize (minimize negative)

         # Define the search space
         search_space = {
             "x": np.arange(-5, 5, 0.1),
             "y": np.arange(-5, 5, 0.1),
         }

         # Create optimizer and solve
         optimizer = HillClimbing(
             search_space=search_space,
             n_iter=100,
             experiment=objective,
         )
         best_params = optimizer.solve()
         print(f"Best parameters: {best_params}")

   .. tab-item:: Scikit-learn Tuning

      .. code-block:: python

         from sklearn.svm import SVC
         from sklearn.datasets import load_iris
         from sklearn.model_selection import train_test_split
         from hyperactive.integrations.sklearn import OptCV
         from hyperactive.opt.gfo import HillClimbing

         # Load data
         X, y = load_iris(return_X_y=True)
         X_train, X_test, y_train, y_test = train_test_split(X, y)

         # Define optimizer with search space
         search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
         optimizer = HillClimbing(search_space=search_space, n_iter=20)

         # Create tuned estimator and fit
         tuned_svc = OptCV(SVC(), optimizer)
         tuned_svc.fit(X_train, y_train)

         print(f"Best params: {tuned_svc.best_params_}")

   .. tab-item:: Bayesian Optimization

      .. code-block:: python

         import numpy as np
         from hyperactive.opt.gfo import BayesianOptimizer

         def complex_objective(params):
             x = params["x"]
             y = params["y"]
             return -((x - 2)**2 + (y + 1)**2) + np.sin(x * y)

         search_space = {
             "x": np.linspace(-5, 5, 100),
             "y": np.linspace(-5, 5, 100),
         }

         optimizer = BayesianOptimizer(
             search_space=search_space,
             n_iter=50,
             experiment=complex_objective,
         )
         best_params = optimizer.solve()

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

Installation
============

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
