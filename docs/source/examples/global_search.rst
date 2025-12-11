.. _examples_global_search:

========================
Global Search Algorithms
========================

Global search algorithms explore the search space more broadly, using
randomization or systematic patterns to avoid getting trapped in local optima.


Algorithm Examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Random Search
     - `random_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/random_search_example.py>`_
   * - Grid Search
     - `grid_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/grid_search_example.py>`_
   * - Random Restart Hill Climbing
     - `random_restart_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/random_restart_hill_climbing_example.py>`_
   * - Stochastic Hill Climbing
     - `stochastic_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/stochastic_hill_climbing_example.py>`_
   * - Powell's Method
     - `powells_method_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/powells_method_example.py>`_
   * - Pattern Search
     - `pattern_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/pattern_search_example.py>`_


When to Use Global Search
-------------------------

Global search algorithms are best suited for:

- **Multimodal search spaces** with multiple local optima
- **Initial exploration** before fine-tuning with local search
- **Unknown search spaces** where the landscape is not well understood
- **Baseline comparisons** (especially Random Search)

See :ref:`user_guide_optimizers` for detailed algorithm descriptions.
