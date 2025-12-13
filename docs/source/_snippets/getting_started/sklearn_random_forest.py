"""Scikit-learn RandomForest example for documentation.

This snippet demonstrates how to optimize a RandomForest classifier
using Hyperactive's SklearnCvExperiment. It is included in get_started.rst.
"""

# [start:full_example]
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
# [end:full_example]

if __name__ == "__main__":
    # Verify we got valid hyperparameters
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params
    print("Sklearn RandomForest example passed!")
