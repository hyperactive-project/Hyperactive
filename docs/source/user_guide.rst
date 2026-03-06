.. _user_guide:

==========
User Guide
==========

Master Hyperactive's optimization toolkit. This guide covers core concepts,
algorithm selection, and integration with popular ML frameworks.

.. tip::

   New to Hyperactive? Start with :ref:`user_guide_introduction` for the fundamentals,
   then explore :ref:`user_guide_optimizers` to choose your algorithm.

----

How Hyperactive Works
---------------------

Hyperactive separates *what* you optimize from *how* you optimize it.
This design lets you swap algorithms without changing your experiment code.

.. raw:: html

   <div class="theme-aware-diagram">
      <img src="_static/diagrams/user_guide_workflow_light.svg"
           alt="Hyperactive workflow: Define experiment and search space, choose optimizer, run solve(), get best parameters"
           class="only-light" />
      <img src="_static/diagrams/user_guide_workflow_dark.svg"
           alt="Hyperactive workflow: Define experiment and search space, choose optimizer, run solve(), get best parameters"
           class="only-dark" />
   </div>

|

Core Concepts
-------------

Every optimization in Hyperactive involves three components:

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: Experiments
      :class-card: sd-border-primary

      **What to optimize**
      ^^^
      Your objective function that takes parameters and returns a score.
      Can be a simple function or a built-in ML experiment class.

      +++
      :doc:`Learn more <user_guide/experiments>`

   .. grid-item-card:: Optimizers
      :class-card: sd-border-success

      **How to optimize**
      ^^^
      The algorithm that explores your search space. Choose from 31 algorithms
      across local search, global search, population methods, and Bayesian approaches.

      +++
      :doc:`Learn more <user_guide/optimizers/index>`

   .. grid-item-card:: Search Spaces
      :class-card: sd-border-warning

      **Where to search**
      ^^^
      Parameter ranges defined as Python dictionaries. Supports categorical,
      integer, and continuous parameters.

      +++
      :doc:`Learn more <user_guide/search_spaces>`

----

Quick Example
-------------

.. code-block:: python

    from hyperactive.opt import HillClimbing

    # 1. Define what to optimize
    def objective(params):
        x, y = params["x"], params["y"]
        return -(x**2 + y**2)  # Minimize x² + y²

    # 2. Define where to search
    search_space = {
        "x": list(range(-10, 11)),
        "y": list(range(-10, 11)),
    }

    # 3. Choose how to optimize & run
    optimizer = HillClimbing(search_space, n_iter=100, experiment=objective)
    best = optimizer.solve()  # Returns {"x": 0, "y": 0}

----

Guide Sections
--------------

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Introduction
      :link: user_guide/introduction
      :link-type: doc

      Core concepts and architecture.
      **Start here** if you're new.

   .. grid-item-card:: Search Spaces
      :link: user_guide/search_spaces
      :link-type: doc

      Best practices for parameter ranges,
      scaling, and granularity.

   .. grid-item-card:: Optimizers
      :link: user_guide/optimizers/index
      :link-type: doc

      Algorithm selection guide.
      31 algorithms across 5 categories.

   .. grid-item-card:: Experiments
      :link: user_guide/experiments
      :link-type: doc

      Custom functions and built-in
      ML experiment classes.

   .. grid-item-card:: Integrations
      :link: user_guide/integrations
      :link-type: doc

      sklearn, sktime, skpro, and
      PyTorch Lightning support.

   .. grid-item-card:: Migration (v4 → v5)
      :link: user_guide/migration
      :link-type: doc

      Upgrade guide with API changes
      and new patterns.


.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/introduction
   user_guide/search_spaces
   user_guide/optimizers/index
   user_guide/experiments
   user_guide/integrations
   user_guide/migration
