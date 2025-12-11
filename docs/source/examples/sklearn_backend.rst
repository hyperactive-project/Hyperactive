.. _examples_sklearn_backend:

===============
Sklearn Backend
===============

Hyperactive provides scikit-learn compatible interfaces that work as drop-in
replacements for ``GridSearchCV`` and ``RandomizedSearchCV``.


Example Files
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Classification with OptCV
     - `sklearn_classif_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sklearn_classif_example.py>`_
   * - Grid Search
     - `grid_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/sklearn/grid_search_example.py>`_
   * - Random Search
     - `random_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/sklearn/random_search_example.py>`_


Usage Overview
--------------

Hyperactive's sklearn-compatible classes follow the familiar fit/predict pattern:

.. code-block:: python

    from hyperactive.integrations.sklearn import HyperactiveSearchCV
    from sklearn.ensemble import RandomForestClassifier

    # Define search space
    param_space = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
    }

    # Create search object
    search = HyperactiveSearchCV(
        estimator=RandomForestClassifier(),
        param_space=param_space,
        n_iter=50,
    )

    # Fit like any sklearn estimator
    search.fit(X_train, y_train)

    # Access best parameters
    print(search.best_params_)

See :ref:`user_guide_integrations` for complete documentation.
