.. _user_guide_optimizers_sequential_model_based:

==============================
Sequential Model-Based Methods
==============================

Sequential model-based optimizers build a model of the objective function and use it
to decide where to sample next. They are best for expensive evaluations where you want
to minimize the number of function calls.

These methods learn from past evaluations to make smarter decisions about which
parameters to try next.


Bayesian Optimization
---------------------

Uses Gaussian Process regression to model the objective and acquisition functions
to balance exploration and exploitation.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:bayesian_optimizer]
   :end-before: # [end:bayesian_optimizer]


Tree-Structured Parzen Estimators (TPE)
---------------------------------------

Models the distribution of good and bad parameters separately.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:tpe]
   :end-before: # [end:tpe]


Forest Optimizer
----------------

Uses Random Forest to model the objective function.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:forest_optimizer]
   :end-before: # [end:forest_optimizer]


Lipschitz Optimization and DIRECT Algorithm
-------------------------------------------

Use Lipschitz continuity assumptions to guide the search.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:lipschitz_direct]
   :end-before: # [end:lipschitz_direct]


When to Use Model-Based Methods
-------------------------------

Sequential model-based optimizers are best suited for:

- **Expensive evaluations**: Training ML models, simulations, real-world experiments
- **Limited budget**: When you can only afford 10-100 evaluations
- **Smooth landscapes**: When the objective function has some continuity

Consider using simpler methods if:

- Evaluations are cheap (random search may find good solutions faster)
- The search space is very large or high-dimensional
- The landscape is highly non-smooth or discontinuous
