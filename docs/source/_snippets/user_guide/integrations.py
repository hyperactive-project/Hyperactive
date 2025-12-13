"""Integrations page code snippets for documentation.

This snippet file contains examples from the integrations.rst page covering
sklearn, sktime, skpro, and PyTorch integrations.
"""

# [start:optcv_basic]
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
# [end:optcv_basic]


# [start:different_optimizers]
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
# [end:different_optimizers]


# [start:pipeline_integration]
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
# [end:pipeline_integration]


# [start:forecasting_optcv]
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
# [end:forecasting_optcv]


# [start:tsc_optcv]
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
# [end:tsc_optcv]


# [start:skpro_experiment]
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
# [end:skpro_experiment]


# [start:pytorch_lightning]
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
# [end:pytorch_lightning]


# --- Runnable test code below ---
if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from hyperactive.integrations.sklearn import OptCV
    from hyperactive.opt.gfo import HillClimbing

    # Test OptCV basic usage
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
    optimizer = HillClimbing(search_space=search_space, n_iter=10)
    tuned_svc = OptCV(SVC(), optimizer)
    tuned_svc.fit(X_train, y_train)
    y_pred = tuned_svc.predict(X_test)

    assert hasattr(tuned_svc, "best_params_")
    assert hasattr(tuned_svc, "best_estimator_")
    assert "kernel" in tuned_svc.best_params_
    assert "C" in tuned_svc.best_params_

    # Test pipeline integration
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC()),
    ])

    search_space = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1],
    }

    optimizer = HillClimbing(search_space=search_space, n_iter=5)
    tuned_pipe = OptCV(pipe, optimizer)
    tuned_pipe.fit(X_train, y_train)

    assert hasattr(tuned_pipe, "best_params_")

    print("Integrations snippets passed!")
