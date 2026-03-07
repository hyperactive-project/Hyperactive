.. _user_guide_optimizers_configuration:

=======================
Optimizer Configuration
=======================

This page covers configuration options shared across all Hyperactive optimizers.


Common Parameters
-----------------

All optimizers accept these parameters:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:common_parameters]
   :end-before: # [end:common_parameters]


Initialization Strategies
-------------------------

Control how the optimizer initializes its search:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:warm_start_example]
   :end-before: # [end:warm_start_example]

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:initialization_strategies]
   :end-before: # [end:initialization_strategies]


Algorithm-Specific Parameters
-----------------------------

Many optimizers have additional parameters. Check the :ref:`api_reference` for details.

Example with Simulated Annealing:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:simulated_annealing_config]
   :end-before: # [end:simulated_annealing_config]


Performance Tips
----------------

1. **Start with baselines**: Always run ``RandomSearch`` first to establish
   a baseline and understand your objective landscape.

2. **Match iterations to complexity**: Complex optimizers (Bayesian, population-based)
   need more iterations to show their advantages.

3. **Consider evaluation cost**: For cheap evaluations, simple optimizers work well.
   For expensive ones, use model-based approaches.

4. **Use warm starts**: If you have prior knowledge, warm start can significantly
   speed up optimization.

5. **Set random seeds**: For reproducible results, always set ``random_state``.
