.. _examples_sequential_model_based:

=================================
Sequential Model-Based Algorithms
=================================

Sequential model-based optimization (SMBO) algorithms build a surrogate model
of the objective function to guide the search. They are particularly effective
when function evaluations are expensive.


Algorithm Examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Bayesian Optimization
     - `bayesian_optimization_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/bayesian_optimization_example.py>`_
   * - Tree-Parzen Estimators
     - `tree_structured_parzen_estimators_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/tree_structured_parzen_estimators_example.py>`_
   * - Forest Optimizer
     - `forest_optimizer_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/forest_optimizer_example.py>`_
   * - Lipschitz Optimizer
     - `lipschitz_optimizer_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/lipschitz_optimizer_example.py>`_
   * - DIRECT Algorithm
     - `direct_algorithm_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/direct_algorithm_example.py>`_


When to Use Model-Based Methods
-------------------------------

Sequential model-based algorithms are best suited for:

- **Expensive objective functions** (e.g., training neural networks, simulations)
- **Limited evaluation budgets** where each evaluation counts
- **Smooth, continuous search spaces** where surrogate models work well
- **Hyperparameter optimization** for machine learning models

See :ref:`user_guide_optimizers` for detailed algorithm descriptions.
