"""Experiments page code snippets for documentation.

This snippet file contains examples from the experiments.rst page covering
custom objectives, built-in experiments, and benchmarks.
"""

import numpy as np

# [start:simple_objective]
def objective(params):
    x = params["x"]
    y = params["y"]
    # Hyperactive MAXIMIZES this score
    return -(x**2 + y**2)
# [end:simple_objective]


# [start:ackley_function]
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
# [end:ackley_function]


# [start:external_simulation]
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
# [end:external_simulation]


# [start:sklearn_cv_experiment]
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
# [end:sklearn_cv_experiment]


# [start:sktime_forecasting]
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
# [end:sktime_forecasting]


# [start:torch_experiment]
from hyperactive.experiment.integrations import TorchExperiment

experiment = TorchExperiment(
    model_class=MyLightningModel,
    datamodule=my_datamodule,
    trainer_kwargs={"max_epochs": 10},
)
# [end:torch_experiment]


# [start:benchmark_experiments]
from hyperactive.experiment.bench import Ackley, Sphere, Parabola

# Use benchmark as experiment
ackley = Ackley(dim=2)

optimizer = BayesianOptimizer(
    search_space=ackley.search_space,
    n_iter=50,
    experiment=ackley,
)
# [end:benchmark_experiments]


# [start:score_method]
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
# [end:score_method]


# [start:robust_objective]
def robust_objective(params):
    try:
        score = compute_score(params)
        return score
    except Exception:
        return -np.inf  # Return bad score on failure
# [end:robust_objective]


# --- Runnable test code below ---
if __name__ == "__main__":
    # Test simple objective
    params = {"x": 0.0, "y": 0.0}
    score = objective(params)
    assert score == 0.0, f"Expected 0.0, got {score}"

    # Test Ackley function
    params = {"x": 0.0, "y": 0.0}
    ackley_score = ackley(params)
    # Ackley minimum is at (0,0) with value 0
    assert abs(ackley_score) < 0.01, f"Expected ~0, got {ackley_score}"

    # Test sklearn CV experiment
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from hyperactive.opt.gfo import HillClimbing

    X, y = load_iris(return_X_y=True)
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        X=X,
        y=y,
        cv=3,
    )

    search_space = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
    }

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=5,
        experiment=experiment,
        random_state=42,
    )
    best_params = optimizer.solve()
    assert "n_estimators" in best_params
    assert "max_depth" in best_params

    print("Experiments snippets passed!")
