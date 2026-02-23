.. _user_guide_optimizers_global_search:

=============
Global Search
=============

Global search optimizers explore the entire search space more thoroughly.
They aim to find good solutions without getting trapped in local optima.


Random Search
-------------

Samples random points from the search space. Simple but surprisingly effective baseline.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:random_search]
   :end-before: # [end:random_search]


Grid Search
-----------

Evaluates all combinations systematically. Only practical for small search spaces.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:grid_search]
   :end-before: # [end:grid_search]


Random Restart Hill Climbing
----------------------------

Runs hill climbing from multiple random starting points.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:random_restart_hill_climbing]
   :end-before: # [end:random_restart_hill_climbing]


Powell's Method and Pattern Search
----------------------------------

Classical derivative-free optimization methods.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:powells_pattern]
   :end-before: # [end:powells_pattern]


When to Use Global Search
-------------------------

Global search optimizers are best suited for:

- **Establishing baselines**: Random search is a strong baseline for any problem
- **Small search spaces**: Grid search provides exhaustive coverage
- **Unknown landscapes**: When you don't know the structure of your objective
- **Simple problems**: When more sophisticated methods aren't necessary

Consider using population-based or model-based methods if:

- Random search isn't finding good solutions
- You have expensive evaluations and need smarter exploration
- Your search space is too large for grid search
