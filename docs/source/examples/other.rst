.. _examples_other:

=================
Advanced Examples
=================

These examples demonstrate advanced Hyperactive features for more sophisticated
optimization workflows.


Warm Starting Optimization
--------------------------

Start optimization from known good points to accelerate convergence:

.. literalinclude:: ../_snippets/examples/advanced_examples.py
   :language: python
   :start-after: # [start:warm_starting]
   :end-before: # [end:warm_starting]


Comparing Optimizers
--------------------

Compare different optimization strategies on the same problem:

.. literalinclude:: ../_snippets/examples/advanced_examples.py
   :language: python
   :start-after: # [start:comparing_optimizers]
   :end-before: # [end:comparing_optimizers]


Tips for Advanced Usage
-----------------------

**Warm Starting**

- Use results from previous runs to seed new optimizations
- Helpful when iterating on model architecture or features
- Combine with local search for fine-tuning around known good points

**Optimizer Comparison**

- Always use the same ``random_state`` for reproducible comparisons
- Run multiple trials to account for optimizer randomness
- Consider both final score and convergence speed
