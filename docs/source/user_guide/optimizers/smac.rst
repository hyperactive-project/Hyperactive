.. _user_guide_optimizers_smac:

============
SMAC Backend
============

Hyperactive provides wrappers for SMAC3's (Sequential Model-based Algorithm
Configuration) Bayesian optimization algorithms, offering state-of-the-art
surrogate model-based optimization.


Available Optimizers
--------------------

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:smac_imports]
   :end-before: # [end:smac_imports]


Example: SmacRandomForest
-------------------------

The flagship SMAC optimizer using a Random Forest surrogate model. Handles mixed
parameter types (continuous, categorical, integer) natively:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:smac_random_forest]
   :end-before: # [end:smac_random_forest]


Example: SmacGaussianProcess
----------------------------

Uses a Gaussian Process surrogate model. Best for continuous parameter spaces
with small to moderate evaluation budgets:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:smac_gaussian_process]
   :end-before: # [end:smac_gaussian_process]


When to Use SMAC Backend
------------------------

The SMAC backend is useful when:

- You need state-of-the-art Bayesian optimization from the AutoML community
- Your search space has mixed parameter types (Random Forest handles these well)
- You want sample-efficient optimization for expensive function evaluations
- You need reproducible results following established AutoML best practices

Choose ``SmacRandomForest`` for mixed parameter spaces and larger budgets (50+).
Choose ``SmacGaussianProcess`` for purely continuous spaces with smaller budgets.
Choose ``SmacRandomSearch`` as a baseline for comparison.
