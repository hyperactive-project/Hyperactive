.. _get_started:

===========
Get Started
===========

This guide will help you get up and running with Hyperactive in just a few minutes.
By the end, you'll understand the core concepts and be able to run your first optimization.

Quick Start
-----------

Hyperactive makes hyperparameter optimization simple. Here's a complete example
that optimizes a custom function:

.. code-block:: python

    import numpy as np
    from hyperactive.opt.gfo import HillClimbing

    # 1. Define your objective function
    def objective(params):
        x = params["x"]
        y = params["y"]
        return -(x**2 + y**2)  # Hyperactive maximizes by default

    # 2. Define the search space
    search_space = {
        "x": np.arange(-5, 5, 0.1),
        "y": np.arange(-5, 5, 0.1),
    }

    # 3. Create an optimizer and solve
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )
    best_params = optimizer.solve()

    print(f"Best parameters: {best_params}")

That's it! Let's break down what happened:

1. **Objective function**: A callable that takes a dictionary of parameters and returns a score.
   Hyperactive **maximizes** this score by default.

2. **Search space**: A dictionary mapping parameter names to their possible values.
   Use NumPy arrays or lists to define discrete search spaces.

3. **Optimizer**: Choose from 20+ optimization algorithms. Each optimizer explores the
   search space differently to find optimal parameters.


First Steps
-----------

Optimizing a Scikit-learn Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common use case is tuning machine learning models. Here's how to optimize
a Random Forest classifier:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from hyperactive.opt.gfo import HillClimbing

    # Load data
    X, y = load_iris(return_X_y=True)

    # Create an experiment that handles cross-validation
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        X=X,
        y=y,
        cv=5,
    )

    # Define hyperparameter search space
    search_space = {
        "n_estimators": list(range(10, 200, 10)),
        "max_depth": list(range(1, 20)),
        "min_samples_split": list(range(2, 10)),
    }

    # Optimize
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=50,
        experiment=experiment,
    )
    best_params = optimizer.solve()

    print(f"Best hyperparameters: {best_params}")


Using the Sklearn Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For even simpler sklearn integration, use the ``OptCV`` wrapper that behaves like
scikit-learn's ``GridSearchCV``:

.. code-block:: python

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from hyperactive.integrations.sklearn import OptCV
    from hyperactive.opt.gfo import HillClimbing

    # Load and split data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define optimizer with search space
    search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100]}
    optimizer = HillClimbing(search_space=search_space, n_iter=20)

    # Create tuned estimator (like GridSearchCV)
    tuned_svc = OptCV(SVC(), optimizer)

    # Fit and predict as usual
    tuned_svc.fit(X_train, y_train)
    y_pred = tuned_svc.predict(X_test)

    # Access results
    print(f"Best params: {tuned_svc.best_params_}")
    print(f"Best estimator: {tuned_svc.best_estimator_}")


Choosing an Optimizer
^^^^^^^^^^^^^^^^^^^^^

Hyperactive provides many optimization algorithms. Here are some common choices:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Optimizer
     - Best For
   * - ``HillClimbing``
     - Quick local optimization, good starting point
   * - ``RandomSearch``
     - Exploring large search spaces, baseline comparison
   * - ``BayesianOptimizer``
     - Expensive evaluations, smart exploration
   * - ``ParticleSwarmOptimizer``
     - Multi-modal problems, avoiding local optima
   * - ``GeneticAlgorithm``
     - Complex landscapes, combinatorial problems

Example with Bayesian Optimization:

.. code-block:: python

    from hyperactive.opt.gfo import BayesianOptimizer

    optimizer = BayesianOptimizer(
        search_space=search_space,
        n_iter=30,
        experiment=experiment,
    )
    best_params = optimizer.solve()


Next Steps
----------

Now that you've seen the basics, explore these topics:

- :ref:`installation` - Detailed installation instructions
- :ref:`user_guide` - In-depth tutorials and concepts
- :ref:`api_reference` - Complete API documentation
- :ref:`examples` - More code examples

Key Concepts to Learn
^^^^^^^^^^^^^^^^^^^^^

1. **Experiments**: Abstractions that define *what* to optimize (see :ref:`user_guide_experiments`)
2. **Optimizers**: Algorithms that define *how* to optimize (see :ref:`user_guide_optimizers`)
3. **Search Spaces**: Define the parameter ranges to explore
4. **Integrations**: Built-in support for sklearn, sktime, and PyTorch (see :ref:`user_guide_integrations`)
