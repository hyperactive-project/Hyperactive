.. _user_guide_optimizers_optuna:

==============
Optuna Backend
==============

Hyperactive provides wrappers for Optuna's powerful samplers, giving you access to
Optuna's algorithms with Hyperactive's unified interface.


Available Optimizers
--------------------

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:optuna_imports]
   :end-before: # [end:optuna_imports]


Example: TPE with Optuna
------------------------

Example with Optuna TPE:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:optuna_tpe]
   :end-before: # [end:optuna_tpe]


When to Use Optuna Backend
--------------------------

The Optuna backend is useful when:

- You want access to Optuna's well-tested sampler implementations
- You're familiar with Optuna and want similar behavior
- You need specific Optuna features like CMA-ES or NSGA-II/III

For most use cases, the native GFO optimizers provide equivalent functionality
with the same interface.
