.. _user_guide_integrations:

======================
Framework Integrations
======================

Hyperactive provides seamless integrations with popular machine learning frameworks.
These integrations offer drop-in replacements for tools like ``GridSearchCV``,
making it easy to use any Hyperactive optimizer with your existing code.


Scikit-Learn Integration
------------------------

The ``OptCV`` class provides a scikit-learn compatible interface for hyperparameter
tuning. It works like ``GridSearchCV`` but supports any Hyperactive optimizer.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from hyperactive.integrations.sklearn import OptCV
    from hyperactive.opt.gfo import HillClimbing

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define search space and optimizer
    search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100]}
    optimizer = HillClimbing(search_space=search_space, n_iter=20)

    # Create tuned estimator
    tuned_svc = OptCV(SVC(), optimizer)

    # Fit like any sklearn estimator
    tuned_svc.fit(X_train, y_train)

    # Predict
    y_pred = tuned_svc.predict(X_test)

    # Access results
    print(f"Best parameters: {tuned_svc.best_params_}")
    print(f"Best estimator: {tuned_svc.best_estimator_}")


Using Different Optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Any Hyperactive optimizer works with ``OptCV``:

.. code-block:: python

    from hyperactive.opt.gfo import BayesianOptimizer, GeneticAlgorithm
    from hyperactive.opt.optuna import TPEOptimizer
    from hyperactive.opt import GridSearchSk as GridSearch

    # Grid Search (exhaustive)
    optimizer = GridSearch(search_space)
    tuned_model = OptCV(SVC(), optimizer)

    # Bayesian Optimization (smart sampling)
    optimizer = BayesianOptimizer(search_space=search_space, n_iter=30)
    tuned_model = OptCV(SVC(), optimizer)

    # Genetic Algorithm (population-based)
    optimizer = GeneticAlgorithm(search_space=search_space, n_iter=50)
    tuned_model = OptCV(SVC(), optimizer)

    # Optuna TPE
    optimizer = TPEOptimizer(search_space=search_space, n_iter=30)
    tuned_model = OptCV(SVC(), optimizer)


Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

``OptCV`` works with sklearn pipelines:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC()),
    ])

    # Search space with pipeline parameter naming
    search_space = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
    }

    optimizer = HillClimbing(search_space=search_space, n_iter=20)
    tuned_pipe = OptCV(pipe, optimizer)
    tuned_pipe.fit(X_train, y_train)


Time Series with Sktime
-----------------------

Hyperactive integrates with ``sktime`` for time series forecasting optimization.

.. note::

   Requires ``pip install hyperactive[sktime-integration]``


Forecasting Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``ForecastingOptCV`` to tune forecasters:

.. code-block:: python

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
    from hyperactive.integrations.sktime import ForecastingOptCV
    from hyperactive.opt import GridSearchSk as GridSearch

    # Load time series data
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    # Define search space
    param_grid = {"strategy": ["mean", "last", "drift"]}

    # Create tuned forecaster
    tuned_forecaster = ForecastingOptCV(
        NaiveForecaster(),
        GridSearch(param_grid),
        cv=ExpandingWindowSplitter(
            initial_window=12,
            step_length=3,
            fh=range(1, 13),
        ),
    )

    # Fit and predict
    tuned_forecaster.fit(y_train, fh=range(1, 13))
    y_pred = tuned_forecaster.predict()

    # Access results
    print(f"Best parameters: {tuned_forecaster.best_params_}")
    print(f"Best forecaster: {tuned_forecaster.best_forecaster_}")


Time Series Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``TSCOptCV`` for time series classification:

.. code-block:: python

    from sktime.classification.dummy import DummyClassifier
    from sktime.datasets import load_unit_test
    from sklearn.model_selection import KFold
    from hyperactive.integrations.sktime import TSCOptCV
    from hyperactive.opt import GridSearchSk as GridSearch

    # Load time series classification data
    X_train, y_train = load_unit_test(
        return_X_y=True,
        split="TRAIN",
        return_type="pd-multiindex",
    )
    X_test, _ = load_unit_test(
        return_X_y=True,
        split="TEST",
        return_type="pd-multiindex",
    )

    # Define search space
    param_grid = {"strategy": ["most_frequent", "stratified"]}

    # Create tuned classifier
    tuned_classifier = TSCOptCV(
        DummyClassifier(),
        GridSearch(param_grid),
        cv=KFold(n_splits=2, shuffle=False),
    )

    # Fit and predict
    tuned_classifier.fit(X_train, y_train)
    y_pred = tuned_classifier.predict(X_test)

    # Access results
    print(f"Best parameters: {tuned_classifier.best_params_}")


Probabilistic Prediction with Skpro
-----------------------------------

For probabilistic regression with ``skpro``:

.. code-block:: python

    from hyperactive.experiment.integrations import SkproProbaRegExperiment
    from hyperactive.opt.gfo import HillClimbing

    experiment = SkproProbaRegExperiment(
        estimator=YourSkproEstimator(),
        X=X,
        y=y,
        cv=5,
    )

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=30,
        experiment=experiment,
    )
    best_params = optimizer.solve()


PyTorch Lightning Integration
-----------------------------

For deep learning hyperparameter optimization with PyTorch Lightning:

.. note::

   Requires ``pip install hyperactive[all_extras]`` or ``pip install lightning``

.. code-block:: python

    from hyperactive.experiment.integrations import TorchExperiment
    from hyperactive.opt.gfo import BayesianOptimizer
    import lightning as L

    # Define your Lightning module
    class MyModel(L.LightningModule):
        def __init__(self, learning_rate=0.001, hidden_size=64):
            super().__init__()
            self.learning_rate = learning_rate
            self.hidden_size = hidden_size
            # ... model definition

        def training_step(self, batch, batch_idx):
            # ... training logic
            pass

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Create experiment
    experiment = TorchExperiment(
        model_class=MyModel,
        datamodule=my_datamodule,
        trainer_kwargs={
            "max_epochs": 10,
            "accelerator": "auto",
        },
    )

    # Define search space
    search_space = {
        "learning_rate": [0.0001, 0.001, 0.01],
        "hidden_size": [32, 64, 128, 256],
    }

    # Optimize
    optimizer = BayesianOptimizer(
        search_space=search_space,
        n_iter=20,
        experiment=experiment,
    )
    best_params = optimizer.solve()


Choosing the Right Integration
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Framework
     - Integration Class
     - Use Case
   * - scikit-learn
     - ``OptCV``
     - Classification, regression, pipelines
   * - sktime
     - ``ForecastingOptCV``
     - Time series forecasting
   * - sktime
     - ``TSCOptCV``
     - Time series classification
   * - skpro
     - ``SkproProbaRegExperiment``
     - Probabilistic regression
   * - PyTorch Lightning
     - ``TorchExperiment``
     - Deep learning models


Tips for Using Integrations
---------------------------

1. **Match the interface**: Use ``OptCV`` when you want sklearn-compatible behavior
   (fit/predict). Use experiment classes when you want more control.

2. **Consider evaluation cost**: Deep learning experiments are expensive.
   Use efficient optimizers like ``BayesianOptimizer`` with fewer iterations.

3. **Use appropriate CV strategies**: Match your cross-validation to your problem
   (e.g., ``TimeSeriesSplit`` for time series, stratified splits for imbalanced data).

4. **Start simple**: Begin with ``GridSearch`` or ``RandomSearch`` to establish
   baselines before using more sophisticated optimizers.
