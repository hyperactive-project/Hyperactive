.. _troubleshooting_results:

====================
Optimization Results
====================

Results Vary Between Runs
-------------------------

**Cause**: Optimization algorithms are stochastic.

**Solution**: Set a random seed for reproducibility:

.. code-block:: python

    optimizer = HillClimbing(
        search_space=space,
        n_iter=100,
        experiment=objective,
        random_state=42,
    )


Optimizer Gets Stuck in Local Optima
------------------------------------

**Cause**: Local search algorithms (like HillClimbing) can get trapped.

**Solutions**:

1. Use a global search algorithm:

   .. code-block:: python

       from hyperactive.opt.gfo import RandomSearch, BayesianOptimizer

2. Use population-based methods:

   .. code-block:: python

       from hyperactive.opt.gfo import ParticleSwarmOptimizer, GeneticAlgorithm

3. Increase exploration in local search:

   .. code-block:: python

       optimizer = HillClimbing(
           search_space=space,
           n_iter=100,
           experiment=objective,
           epsilon=0.2,  # Larger steps
       )


Best Score is Very Low or Negative
----------------------------------

**Check these**:

1. **Objective function errors** — Make sure your objective doesn't crash:

   .. code-block:: python

       def objective(params):
           try:
               score = evaluate(params)
               return score
           except Exception as e:
               print(f"Error: {e}")  # Debug
               return -np.inf

2. **Sign convention** — Hyperactive maximizes. Negate if minimizing:

   .. code-block:: python

       def objective(params):
           error = compute_error(params)
           return -error  # Negate for minimization
