.. _faq_common_issues:

=============
Common Issues
=============

Why is my optimization slow?
----------------------------

**Slow objective function**: The optimizer only controls search strategy.
If each evaluation takes a long time, consider:

- Reducing cross-validation folds
- Using a subset of training data for tuning
- Simplifying your model during search

**Large search space**: More combinations require more iterations.
Consider reducing parameter granularity or using smarter optimizers
like Bayesian optimization.

**Too many iterations**: Start with fewer iterations and increase
if needed.


Why does my score vary between runs?
------------------------------------

Optimization algorithms are stochastic. To get reproducible results,
set a random seed:

.. code-block:: python

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
        random_state=42,  # Set seed for reproducibility
    )


My objective function returns NaN or raises exceptions
------------------------------------------------------

Handle invalid configurations in your objective function:

.. code-block:: python

    def objective(params):
        try:
            score = evaluate_model(params)
            if np.isnan(score):
                return -np.inf  # Return worst possible score
            return score
        except Exception:
            return -np.inf  # Return worst possible score on error


How do I see what parameters were tried?
----------------------------------------

Access the search history after optimization:

.. code-block:: python

    best_params = optimizer.solve()

    # Access results
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best score: {optimizer.best_score_}")

    # Full search history (if available)
    # Check optimizer attributes for search_data or similar
