.. _faq_advanced_usage:

==============
Advanced Usage
==============

Can I run optimizations in parallel?
------------------------------------

Currently, Hyperactive v5 runs single optimizer instances.
For parallel evaluation of candidates, consider using Optuna
backend optimizers which support parallel trials:

.. code-block:: python

    from hyperactive.opt.optuna import TPEOptimizer

    optimizer = TPEOptimizer(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
        # Optuna handles parallelization
    )


Can I save and resume optimization?
-----------------------------------

This feature is planned but not yet available in v5. As a workaround,
you can log results during optimization and use them as initial points
for a new run.


Are callbacks supported?
------------------------

User-defined callbacks during optimization are not currently supported in v5.
The Optuna backend has internal early-stopping callbacks, but there's no
general callback interface for tracking progress or modifying behavior during
optimization.

For progress monitoring, you can add logging inside your objective function:

.. code-block:: python

    iteration = 0

    def objective(params):
        global iteration
        iteration += 1
        score = evaluate_model(params)
        print(f"Iteration {iteration}: score={score:.4f}")
        return score


How do I add constraints between parameters?
--------------------------------------------

Handle constraints in your objective function by returning a poor score
for invalid combinations:

.. code-block:: python

    def objective(params):
        # Constraint: min_samples_split must be >= min_samples_leaf
        if params["min_samples_split"] < params["min_samples_leaf"]:
            return -np.inf  # Invalid configuration

        return evaluate_model(params)
