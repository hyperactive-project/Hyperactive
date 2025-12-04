.. _user_guide_experiments:

===========
Experiments
===========

Experiments define *what* to optimize in Hyperactive. They encapsulate the objective
function and any evaluation logic needed to score a set of parameters.

Defining Experiments
--------------------

There are two ways to define experiments in Hyperactive:

1. **Custom functions** — Simple callables for any optimization problem
2. **Built-in experiment classes** — Pre-built experiments for common ML tasks


Custom Objective Functions
--------------------------

The simplest way to define an experiment is as a Python function that takes
a parameter dictionary and returns a score:

.. code-block:: python

    def objective(params):
        x = params["x"]
        y = params["y"]
        # Hyperactive MAXIMIZES this score
        return -(x**2 + y**2)

Key points:

- The function receives a dictionary with parameter names as keys
- It must return a single numeric value (the score)
- Hyperactive **maximizes** this score by default
- To minimize, negate your loss function (as shown above)


Example: Optimizing a Mathematical Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from hyperactive.opt.gfo import BayesianOptimizer

    # Ackley function (a common benchmark)
    def ackley(params):
        x = params["x"]
        y = params["y"]

        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        result = term1 + term2 + np.e + 20

        return -result  # Negate to maximize (minimize the Ackley function)

    search_space = {
        "x": np.linspace(-5, 5, 100),
        "y": np.linspace(-5, 5, 100),
    }

    optimizer = BayesianOptimizer(
        search_space=search_space,
        n_iter=50,
        experiment=ackley,
    )
    best_params = optimizer.solve()


Example: Optimizing with External Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your objective function can use any Python code:

.. code-block:: python

    import subprocess

    def run_simulation(params):
        # Run an external simulation with the given parameters
        result = subprocess.run(
            ["./my_simulation", str(params["param1"]), str(params["param2"])],
            capture_output=True,
            text=True,
        )
        # Parse the output and return the score
        score = float(result.stdout.strip())
        return score


Built-in Experiment Classes
---------------------------

For common machine learning tasks, Hyperactive provides ready-to-use experiment classes
that handle cross-validation, scoring, and other details.


SklearnCvExperiment
^^^^^^^^^^^^^^^^^^^

For optimizing scikit-learn estimators with cross-validation:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from hyperactive.opt.gfo import HillClimbing

    X, y = load_iris(return_X_y=True)

    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        X=X,
        y=y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=accuracy_score,  # Optional: defaults to estimator's score method
    )

    search_space = {
        "n_estimators": list(range(10, 200, 10)),
        "max_depth": list(range(1, 20)),
        "min_samples_split": list(range(2, 10)),
    }

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=30,
        experiment=experiment,
    )
    best_params = optimizer.solve()


SktimeForecastingExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For time series forecasting optimization (requires ``sktime``):

.. code-block:: python

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.datasets import load_airline
    from hyperactive.experiment.integrations import SktimeForecastingExperiment
    from hyperactive.opt.gfo import RandomSearch

    y = load_airline()

    experiment = SktimeForecastingExperiment(
        estimator=NaiveForecaster(),
        y=y,
        fh=[1, 2, 3],  # Forecast horizon
    )

    search_space = {
        "strategy": ["mean", "last", "drift"],
    }

    optimizer = RandomSearch(
        search_space=search_space,
        n_iter=10,
        experiment=experiment,
    )
    best_params = optimizer.solve()


TorchExperiment
^^^^^^^^^^^^^^^

For PyTorch Lightning model optimization (requires ``lightning``):

.. code-block:: python

    from hyperactive.experiment.integrations import TorchExperiment

    experiment = TorchExperiment(
        model_class=MyLightningModel,
        datamodule=my_datamodule,
        trainer_kwargs={"max_epochs": 10},
    )


Benchmark Experiments
---------------------

Hyperactive includes standard benchmark functions for testing optimizers:

.. code-block:: python

    from hyperactive.experiment.bench import Ackley, Sphere, Parabola

    # Use benchmark as experiment
    ackley = Ackley(dim=2)

    optimizer = BayesianOptimizer(
        search_space=ackley.search_space,
        n_iter=50,
        experiment=ackley,
    )


Using the score() Method
------------------------

Experiments can also be evaluated directly using the ``score()`` method:

.. code-block:: python

    from hyperactive.experiment.integrations import SklearnCvExperiment

    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(),
        X=X, y=y, cv=5,
    )

    # Evaluate specific parameters
    params = {"n_estimators": 100, "max_depth": 10}
    score, additional_info = experiment.score(params)

    print(f"Score: {score}")
    print(f"Additional info: {additional_info}")


Tips for Designing Experiments
------------------------------

1. **Return meaningful scores**: Ensure your score reflects what you want to optimize.
   Higher is better (Hyperactive maximizes).

2. **Handle errors gracefully**: If a parameter combination fails, return a very
   low score (e.g., ``-np.inf``) rather than raising an exception.

3. **Consider computation time**: For expensive experiments, use efficient optimizers
   like ``BayesianOptimizer`` that learn from previous evaluations.

4. **Use reproducibility**: Set random seeds in your experiment for consistent results.

.. code-block:: python

    def robust_objective(params):
        try:
            score = compute_score(params)
            return score
        except Exception:
            return -np.inf  # Return bad score on failure
