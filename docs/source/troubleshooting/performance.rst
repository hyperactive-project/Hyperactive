.. _troubleshooting_performance:

==================
Performance Issues
==================

Optimization is Very Slow
-------------------------

**Possible causes and solutions**:

1. **Slow objective function**

   The optimizer can only be as fast as your objective. Consider:

   - Reducing cross-validation folds (``cv=3`` instead of ``cv=10``)
   - Using a subset of data during tuning
   - Using a simpler model for initial exploration

2. **Too many iterations**

   Start with fewer iterations:

   .. code-block:: python

       optimizer = HillClimbing(
           search_space=space,
           n_iter=50,  # Start small
           experiment=objective,
       )

3. **Overly large search space**

   Reduce granularity or the number of parameters:

   .. code-block:: python

       # Instead of 1000 values
       "learning_rate": np.linspace(0.001, 0.1, 1000)

       # Use 20-50 values
       "learning_rate": np.logspace(-3, -1, 20)


Memory Errors
-------------

**Cause**: Very large search spaces can cause memory issues with some optimizers,
especially those that cache all combinations.

**Solution**:

1. Reduce search space size
2. Use sampling-based optimizers (``RandomSearch``, ``BayesianOptimizer``)
3. Use coarser parameter granularity

.. code-block:: python

    # High memory usage
    search_space = {
        "a": np.linspace(0, 1, 10000),
        "b": np.linspace(0, 1, 10000),
    }  # 100 million combinations!

    # Lower memory usage
    search_space = {
        "a": np.linspace(0, 1, 100),
        "b": np.linspace(0, 1, 100),
    }  # 10,000 combinations
