.. _troubleshooting_runtime:

==============
Runtime Errors
==============

AttributeError: 'X' object has no attribute 'run'
-------------------------------------------------

**Cause**: Using v4 method names with v5 optimizers.

**Solution**: Use ``.solve()`` instead of ``.run()``:

.. code-block:: python

    # Old (v4)
    hyper.run()

    # New (v5)
    best_params = optimizer.solve()


TypeError: unexpected keyword argument
--------------------------------------

**Cause**: Parameter passing changed in v5. All configuration now goes
to the optimizer constructor.

**Solution**:

.. code-block:: python

    # Old (v4)
    hyper.add_search(model, space, optimizer=opt, n_iter=100)

    # New (v5)
    optimizer = HillClimbing(
        search_space=space,
        n_iter=100,
        experiment=objective,
    )


ValueError: Parameters do not match
-----------------------------------

**Cause**: Your search space keys don't match what the experiment expects.

**Solution**: Ensure search space keys match the parameters your objective
function or experiment expects:

.. code-block:: python

    # Search space defines "learning_rate"
    search_space = {"learning_rate": [0.01, 0.1]}

    # Objective must use the same key
    def objective(params):
        lr = params["learning_rate"]  # Not "lr" or "LearningRate"
        ...
