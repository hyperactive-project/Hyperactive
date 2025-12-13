.. _examples_local_search:

=======================
Local Search Algorithms
=======================

Local search algorithms explore the search space by making small, incremental
moves from the current position. They are efficient for finding local optima
but may get stuck without escaping mechanisms.


Algorithm Examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Hill Climbing
     - `hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/hill_climbing_example.py>`_
   * - Repulsing Hill Climbing
     - `repulsing_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/repulsing_hill_climbing_example.py>`_
   * - Simulated Annealing
     - `simulated_annealing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/simulated_annealing_example.py>`_
   * - Downhill Simplex
     - `downhill_simplex_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/downhill_simplex_example.py>`_


When to Use Local Search
------------------------

Local search algorithms are best suited for:

- **Smooth search spaces** where nearby points have similar scores
- **Fine-tuning** around a known good region
- **Fast convergence** when a good starting point is available
- **Limited computational budget** where few evaluations are possible

See :ref:`user_guide_optimizers` for detailed algorithm descriptions.
