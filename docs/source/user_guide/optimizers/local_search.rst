.. _user_guide_optimizers_local_search:

============
Local Search
============

Local search optimizers explore the neighborhood of the current best solution.
They are fast but may get stuck in local optima.

These optimizers work by iteratively improving from a starting point, making small
adjustments to find better solutions nearby.


Hill Climbing
-------------

The simplest local search: always move to a better neighbor.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:hill_climbing]
   :end-before: # [end:hill_climbing]


Simulated Annealing
-------------------

Like hill climbing, but sometimes accepts worse solutions to escape local optima.
The "temperature" controls exploration vs exploitation.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:simulated_annealing]
   :end-before: # [end:simulated_annealing]


Stochastic Hill Climbing
------------------------

Hill climbing with a probability of accepting worse solutions. The ``p_accept``
parameter controls exploration: higher values make it more likely to accept
non-improving moves, helping escape local optima.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:stochastic_hill_climbing]
   :end-before: # [end:stochastic_hill_climbing]


Repulsing Hill Climbing
-----------------------

Remembers visited regions and avoids them, encouraging broader exploration.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:repulsing_hill_climbing]
   :end-before: # [end:repulsing_hill_climbing]


Downhill Simplex (Nelder-Mead)
------------------------------

Uses a simplex of points to navigate the search space. Good for continuous problems.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:downhill_simplex]
   :end-before: # [end:downhill_simplex]


When to Use Local Search
------------------------

Local search optimizers are best suited for:

- **Quick initial exploration**: Get a baseline result fast
- **Smooth, unimodal landscapes**: When there's a single optimum to find
- **Refinement**: Fine-tune solutions found by global methods
- **Limited budget**: When you can only afford a few evaluations

Consider using global or population-based methods if:

- Your landscape has many local optima
- You're not finding good solutions with local search
- You need more robust exploration
