.. _user_guide:

==========
User Guide
==========

This guide covers Hyperactive's core concepts and features in depth.
Whether you're new to hyperparameter optimization or an experienced practitioner,
you'll find detailed explanations and practical examples here.

.. note::

   Some code snippets in this guide are **illustrative** and may contain
   placeholders (like ``score`` or ``SomeOptimizer``). For complete, runnable
   examples, see the :ref:`examples` or :ref:`get_started` sections.

.. toctree::
   :maxdepth: 1

   user_guide/introduction
   user_guide/search_spaces
   user_guide/optimizers
   user_guide/experiments
   user_guide/integrations
   user_guide/migration


Overview
--------

Hyperactive v5 introduces a clean **experiment-based architecture** that separates
optimization algorithms from optimization problems:

- **Experiments** define *what* to optimize — the objective function and evaluation logic
- **Optimizers** define *how* to optimize — the search strategy and algorithm

This design allows you to:

- Mix and match any optimizer with any experiment type
- Create reusable experiment definitions for common ML tasks
- Easily switch between different optimization strategies
- Build complex optimization workflows with consistent interfaces

Basic Workflow
^^^^^^^^^^^^^^

Every Hyperactive optimization follows this pattern:

.. code-block:: python

    from hyperactive.opt.gfo import HillClimbing

    # 1. Define the experiment (what to optimize)
    def objective(params):
        return score  # Hyperactive maximizes this

    # 2. Define the search space
    search_space = {
        "param1": [value1, value2, ...],
        "param2": [value1, value2, ...],
    }

    # 3. Choose an optimizer (how to optimize)
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

    # 4. Run the optimization
    best_params = optimizer.solve()


Guide Contents
--------------

:ref:`user_guide_introduction`
    Core concepts: optimizers, experiments, and search spaces.
    Start here to understand Hyperactive's architecture.

:ref:`user_guide_search_spaces`
    Best practices for designing search spaces.
    Covers scaling, granularity, and common patterns.

:ref:`user_guide_optimizers`
    Detailed guide to choosing and configuring optimizers.
    Covers local search, global search, population methods, and Bayesian approaches.

:ref:`user_guide_experiments`
    How to define optimization problems using experiments.
    Includes custom functions and built-in ML experiments.

:ref:`user_guide_integrations`
    Framework integrations for scikit-learn, sktime, skpro, and PyTorch.
    Drop-in replacements for GridSearchCV and similar tools.

:ref:`user_guide_migration`
    Migration guide for upgrading from Hyperactive v4 to v5.
    Covers API changes, new patterns, and troubleshooting.
