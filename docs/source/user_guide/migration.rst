.. _user_guide_migration:

======================
Migration Guide (v4→v5)
======================

This guide helps you migrate from Hyperactive v4 to v5. Version 5 introduces
a new experiment-based architecture with a simplified API.

.. note::

   If you're still using Hyperactive v4 and need documentation for that version,
   see the `Legacy Documentation (v4) <https://simonblanke.github.io/hyperactive-documentation/5.0/>`_.

Quick Summary
-------------

The main changes in v5 are:

1. **No more ``Hyperactive`` class** — Optimizers are used directly
2. **``.run()`` replaced with ``.solve()``** — Single method for optimization
3. **New import paths** — ``hyperactive.opt.gfo`` instead of ``hyperactive.optimizers``
4. **Experiment abstraction** — Built-in ML experiments for scikit-learn, sktime, etc.
5. **Constructor-based configuration** — All parameters passed to optimizer constructor


Basic Migration
---------------

Here's how to convert a v4 script to v5:

**v4 (Old)**

.. code-block:: python

    from hyperactive import Hyperactive
    from hyperactive.optimizers import HillClimbingOptimizer

    def model(opt):
        # Use opt["param"] to access parameters
        score = -(opt["x"] ** 2)
        return score

    search_space = {
        "x": list(range(-10, 10)),
    }

    # Create optimizer separately
    optimizer = HillClimbingOptimizer(epsilon=0.1)

    # Create Hyperactive instance and add search
    hyper = Hyperactive()
    hyper.add_search(
        model,
        search_space,
        optimizer=optimizer,
        n_iter=100,
    )
    hyper.run()

    # Access results
    best_params = hyper.best_para(model)
    best_score = hyper.best_score(model)

**v5 (New)**

.. code-block:: python

    from hyperactive.opt.gfo import HillClimbing

    def objective(params):
        # Use params["param"] to access parameters
        score = -(params["x"] ** 2)
        return score

    search_space = {
        "x": list(range(-10, 10)),
    }

    # Configure optimizer directly
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
        epsilon=0.1,
    )

    # Run optimization
    best_params = optimizer.solve()

    # Access results
    best_score = optimizer.best_score_


Import Path Changes
-------------------

The optimizer imports have changed:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v4 Import
     - v5 Import
   * - ``from hyperactive import Hyperactive``
     - Not needed
   * - ``from hyperactive.optimizers import HillClimbingOptimizer``
     - ``from hyperactive.opt.gfo import HillClimbing``
   * - ``from hyperactive.optimizers import RandomSearchOptimizer``
     - ``from hyperactive.opt.gfo import RandomSearch``
   * - ``from hyperactive.optimizers import BayesianOptimizer``
     - ``from hyperactive.opt.gfo import BayesianOptimizer``

.. tip::

   In v5, you can also import optimizers directly from ``hyperactive.opt``:

   .. code-block:: python

       from hyperactive.opt import HillClimbing, RandomSearch, BayesianOptimizer


Optimizer Name Changes
----------------------

Some optimizer class names have changed:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v4 Name
     - v5 Name
   * - ``HillClimbingOptimizer``
     - ``HillClimbing``
   * - ``RandomSearchOptimizer``
     - ``RandomSearch``
   * - ``ParticleSwarmOptimizer``
     - ``ParticleSwarmOptimizer`` (unchanged)
   * - ``BayesianOptimizer``
     - ``BayesianOptimizer`` (unchanged)
   * - ``TreeStructuredParzenEstimators``
     - ``TreeStructuredParzenEstimators`` (unchanged)


Method Changes
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Action
     - v4
     - v5
   * - Run optimization
     - ``hyper.run()``
     - ``optimizer.solve()``
   * - Get best params
     - ``hyper.best_para(model)``
     - ``optimizer.best_params_``
   * - Get best score
     - ``hyper.best_score(model)``
     - ``optimizer.best_score_``


Scikit-learn Integration
------------------------

v5 introduces experiment classes for cleaner ML integration.

**v4 (Old)**

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from hyperactive import Hyperactive
    from hyperactive.optimizers import HillClimbingOptimizer

    X, y = load_iris(return_X_y=True)

    def model(opt):
        clf = RandomForestClassifier(
            n_estimators=opt["n_estimators"],
            max_depth=opt["max_depth"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
    }

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50)
    hyper.run()

**v5 (New) — Using SklearnCvExperiment**

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from hyperactive.opt.gfo import HillClimbing

    X, y = load_iris(return_X_y=True)

    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(),
        X=X, y=y, cv=3,
    )

    search_space = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
    }

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=50,
        experiment=experiment,
    )
    best_params = optimizer.solve()

**v5 (Alternative) — Using custom function**

The v4-style custom function still works in v5:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from hyperactive.opt.gfo import HillClimbing

    X, y = load_iris(return_X_y=True)

    def objective(params):
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
    }

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=50,
        experiment=objective,
    )
    best_params = optimizer.solve()


New Features in v5
------------------

v5 introduces several new features:

**Experiment Classes**

Pre-built experiments for common ML tasks:

- ``SklearnCvExperiment`` — scikit-learn cross-validation
- ``SktimeForecastingExperiment`` — sktime forecasting
- ``SktimeClassificationExperiment`` — sktime time series classification
- ``SkproProbaRegExperiment`` — skpro probabilistic regression
- ``TorchExperiment`` — PyTorch Lightning models

**Optuna Backend**

Access Optuna optimizers through Hyperactive:

.. code-block:: python

    from hyperactive.opt.optuna import TPEOptimizer, CmaEsOptimizer

**sklearn-Compatible Interface**

Drop-in replacement for ``GridSearchCV``:

.. code-block:: python

    from hyperactive.integrations.sklearn import OptCV

    search = OptCV(estimator=clf, param_space=space, n_iter=50)
    search.fit(X, y)


Removed Features
----------------

The following v4 features are no longer available in v5:

- ``Hyperactive.add_search()`` — Use optimizer constructor instead
- ``Hyperactive.run()`` — Use ``optimizer.solve()`` instead
- ``search_data`` parameter — Data collection handled differently
- ``memory`` parameter — Memory features restructured
- Multiple parallel searches — Use separate optimizer instances


Troubleshooting Migration
-------------------------

**ImportError: cannot import name 'Hyperactive'**

The ``Hyperactive`` class no longer exists. Use optimizers directly:

.. code-block:: python

    # Old
    from hyperactive import Hyperactive

    # New
    from hyperactive.opt.gfo import HillClimbing

**AttributeError: 'HillClimbing' object has no attribute 'run'**

The ``.run()`` method is now ``.solve()``:

.. code-block:: python

    # Old
    hyper.run()

    # New
    best_params = optimizer.solve()

**TypeError: unexpected keyword argument 'optimizer'**

Optimizer parameters are now passed to the constructor:

.. code-block:: python

    # Old
    hyper.add_search(model, space, optimizer=opt, n_iter=100)

    # New
    optimizer = HillClimbing(
        search_space=space,
        n_iter=100,
        experiment=model,
    )


Getting Help
------------

If you encounter issues migrating:

1. Check the :ref:`api_reference` for current API
2. See :ref:`examples` for v5 code examples
3. Open an issue on `GitHub <https://github.com/SimonBlanke/Hyperactive/issues>`_
4. Ask on `Discord <https://discord.gg/7uKdHfdcJG>`_
