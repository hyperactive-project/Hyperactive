.. _home:

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <h1 class="hero-title">Hyperactive</h1>
         <p class="hero-tagline">A unified interface for optimization algorithms and experiments</p>
      </div>
   </div>

   <!-- Stats are auto-counted in conf.py: n_algorithms, n_backends, n_integrations -->
   <div class="stats-strip">
      <div class="stat-item">
         <div class="stat-value" id="stat-algorithms">31</div>
         <div class="stat-label">Algorithms</div>
      </div>
      <div class="stat-item">
         <div class="stat-value" id="stat-backends">3</div>
         <div class="stat-label">Backends</div>
      </div>
      <div class="stat-item">
         <div class="stat-value" id="stat-integrations">5</div>
         <div class="stat-label">Integrations</div>
      </div>
      <div class="stat-item">
         <div class="stat-value">1</div>
         <div class="stat-label">Unified API</div>
      </div>
   </div>


Hyperactive provides a collection of optimization algorithms, accessible through a unified
experiment-based interface that separates optimization problems from algorithms. The library
provides native implementations of algorithms from the Gradient-Free-Optimizers package
alongside direct interfaces to Optuna and scikit-learn optimizers.

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


----

.. _features:

Features
================

What makes Hyperactive stand out for optimization tasks.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card::
      :class-card: feature-card

      **20+ Optimization Algorithms**
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

Optimization Backends
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

Hyperactive works seamlessly with popular machine learning frameworks.

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

.. raw:: html

   <div class="segmented-tabs" id="install-tabs">
      <nav class="segmented-tabs-nav" role="tablist">
         <button class="segmented-tab-btn active" role="tab" aria-selected="true" aria-controls="install-panel-pip">
            pip install
         </button>
         <button class="segmented-tab-btn" role="tab" aria-selected="false" aria-controls="install-panel-extras">
            All Extras
         </button>
         <button class="segmented-tab-btn" role="tab" aria-selected="false" aria-controls="install-panel-sklearn">
            sklearn
         </button>
         <button class="segmented-tab-btn" role="tab" aria-selected="false" aria-controls="install-panel-sktime">
            sktime
         </button>
      </nav>
      <div class="segmented-tabs-content">
         <div class="segmented-tab-panel active" id="install-panel-pip" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install hyperactive</pre></div>
         </div>
         <div class="segmented-tab-panel" id="install-panel-extras" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install hyperactive[all_extras]</pre></div>
         </div>
         <div class="segmented-tab-panel" id="install-panel-sklearn" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install hyperactive[sklearn-integration]</pre></div>
         </div>
         <div class="segmented-tab-panel" id="install-panel-sktime" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install hyperactive[sktime-integration]</pre></div>
         </div>
      </div>
   </div>

----

Quick Example
=============

Get started in just a few lines of code:

.. raw:: html

   <div class="vertical-tabs" id="example-tabs">
      <nav class="vertical-tabs-nav" role="tablist">
         <button class="vertical-tab-btn active" role="tab" aria-selected="true" aria-controls="example-panel-custom" data-tab="example-custom">
            <span class="tab-indicator"></span>
            <span>Custom Function</span>
         </button>
         <button class="vertical-tab-btn" role="tab" aria-selected="false" aria-controls="example-panel-sklearn" data-tab="example-sklearn">
            <span class="tab-indicator"></span>
            <span>Scikit-learn Tuning</span>
         </button>
         <button class="vertical-tab-btn" role="tab" aria-selected="false" aria-controls="example-panel-bayesian" data-tab="example-bayesian">
            <span class="tab-indicator"></span>
            <span>Bayesian Optimization</span>
         </button>
      </nav>
      <div class="vertical-tabs-content">
         <div class="vertical-tab-panel active" id="example-panel-custom" role="tabpanel">

.. literalinclude:: _snippets/getting_started/index_custom_function.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]

.. raw:: html

         </div>
         <div class="vertical-tab-panel" id="example-panel-sklearn" role="tabpanel">

.. literalinclude:: _snippets/getting_started/index_sklearn_tuning.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]

.. raw:: html

         </div>
         <div class="vertical-tab-panel" id="example-panel-bayesian" role="tabpanel">

.. literalinclude:: _snippets/getting_started/index_bayesian.py
   :language: python
   :start-after: # [start:full_example]
   :end-before: # [end:full_example]

.. raw:: html

         </div>
      </div>
   </div>

   <script>
   document.addEventListener('DOMContentLoaded', function() {
      // Handle vertical tabs (Quick Example)
      document.querySelectorAll('.vertical-tabs').forEach(function(tabContainer) {
         const buttons = tabContainer.querySelectorAll('.vertical-tab-btn');
         const panels = tabContainer.querySelectorAll('.vertical-tab-panel');

         buttons.forEach(function(button) {
            button.addEventListener('click', function() {
               buttons.forEach(btn => {
                  btn.classList.remove('active');
                  btn.setAttribute('aria-selected', 'false');
               });
               panels.forEach(panel => panel.classList.remove('active'));

               this.classList.add('active');
               this.setAttribute('aria-selected', 'true');

               const panelId = this.getAttribute('aria-controls');
               const panel = document.getElementById(panelId);
               if (panel) {
                  panel.classList.add('active');
               }
            });
         });
      });

      // Handle segmented tabs (Quick Install)
      document.querySelectorAll('.segmented-tabs').forEach(function(tabContainer) {
         const buttons = tabContainer.querySelectorAll('.segmented-tab-btn');
         const panels = tabContainer.querySelectorAll('.segmented-tab-panel');

         buttons.forEach(function(button) {
            button.addEventListener('click', function() {
               buttons.forEach(btn => {
                  btn.classList.remove('active');
                  btn.setAttribute('aria-selected', 'false');
               });
               panels.forEach(panel => panel.classList.remove('active'));

               this.classList.add('active');
               this.setAttribute('aria-selected', 'true');

               const panelId = this.getAttribute('aria-controls');
               const panel = document.getElementById(panelId);
               if (panel) {
                  panel.classList.add('active');
               }
            });
         });
      });

      // Back to Top - inject into sidebar
      const sidebar = document.querySelector('.bd-toc-nav.page-toc');
      if (sidebar) {
         const backToTopDiv = document.createElement('div');
         backToTopDiv.className = 'back-to-top-sidebar';
         backToTopDiv.innerHTML = '<a href="#">↑ Back to top</a>';
         sidebar.appendChild(backToTopDiv);

         backToTopDiv.querySelector('a').addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
         });

         window.addEventListener('scroll', function() {
            if (window.scrollY > 400) {
               backToTopDiv.classList.add('visible');
            } else {
               backToTopDiv.classList.remove('visible');
            }
         });
      }
   });
   </script>

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

.. raw:: html

   <div class="contents-grid">
      <a href="get_started.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Get Started</div>
               <div class="contents-card-desc">Quick introduction to Hyperactive</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
      <a href="installation.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Installation</div>
               <div class="contents-card-desc">Setup guide and requirements</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
      <a href="user_guide.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">User Guide</div>
               <div class="contents-card-desc">In-depth tutorials and explanations</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
      <a href="api_reference.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">API Reference</div>
               <div class="contents-card-desc">Technical reference for all classes</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
      <a href="examples.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Examples</div>
               <div class="contents-card-desc">Code examples and use cases</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
      <a href="get_involved.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Get Involved</div>
               <div class="contents-card-desc">Contribute to Hyperactive</div>
            </div>
            <div class="contents-card-arrow">→</div>
         </div>
      </a>
   </div>
