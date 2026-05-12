"""
Optimizer Comparison: HillClimbing vs BayesianOptimizer
========================================================
This example demonstrates one of Hyperactive's core strengths:
you can swap optimizers without changing any experiment code.

The same RandomForest experiment and search space is run with two
different optimizers. Only the optimizer line changes — everything
else stays identical.

Dataset : Iris (150 samples, 3 classes, 4 features)
Model   : RandomForestClassifier
Metric  : 5-fold cross-validation accuracy (mean)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import BayesianOptimizer, HillClimbing

# ── 1. Data ──────────────────────────────────────────────────────────────────

X, y = load_iris(return_X_y=True)

# ── 2. Experiment (shared by both optimizers) ─────────────────────────────────
# SklearnCvExperiment wraps cross-validation so you don't have to write
# the CV loop manually. The optimizer will call this internally on each
# candidate set of hyperparameters.

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    scoring=accuracy_score,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    X=X,
    y=y,
)

# ── 3. Search space (shared by both optimizers) ───────────────────────────────

search_space = {
    "n_estimators": list(range(10, 200, 10)),  # 10, 20, ..., 190 trees
    "max_depth": list(range(1, 20)),            # tree depth: 1 to 19
    "min_samples_split": list(range(2, 10)),    # min samples to split: 2 to 9
}

# ── 4. Run with HillClimbing ──────────────────────────────────────────────────
# HillClimbing starts from a random point and moves to neighbouring
# parameter values that improve the score. Fast but can get stuck in
# local optima.

print("=" * 55)
print("Optimizer 1: HillClimbing")
print("=" * 55)

optimizer_hc = HillClimbing(
    search_space=search_space,
    n_iter=30,
    experiment=experiment,
)
best_params_hc = optimizer_hc.solve()

print(f"Best parameters : {best_params_hc}")
print(f"Best CV score   : {optimizer_hc.best_score:.4f}\n")

# ── 5. Run with BayesianOptimizer ─────────────────────────────────────────────
# BayesianOptimizer builds a probabilistic model of the search space
# and uses it to pick the most promising candidates next. Smarter
# exploration — especially useful when n_iter is limited.
#
# Notice: the experiment and search_space are identical to above.
# Only this line changes compared to the HillClimbing run.

print("=" * 55)
print("Optimizer 2: BayesianOptimizer")
print("=" * 55)

optimizer_bo = BayesianOptimizer(
    search_space=search_space,
    n_iter=30,
    experiment=experiment,
)
best_params_bo = optimizer_bo.solve()

print(f"Best parameters : {best_params_bo}")
print(f"Best CV score   : {optimizer_bo.best_score:.4f}\n")

# ── 6. Side-by-side comparison ────────────────────────────────────────────────

print("=" * 55)
print("Comparison")
print("=" * 55)
print(f"{'Optimizer':<25} {'n_estimators':<15} {'max_depth':<12} {'CV Score'}")
print("-" * 55)
print(
    f"{'HillClimbing':<25}"
    f"{best_params_hc['n_estimators']:<15}"
    f"{best_params_hc['max_depth']:<12}"
    f"{optimizer_hc.best_score:.4f}"
)
print(
    f"{'BayesianOptimizer':<25}"
    f"{best_params_bo['n_estimators']:<15}"
    f"{best_params_bo['max_depth']:<12}"
    f"{optimizer_bo.best_score:.4f}"
)

# ── Key takeaway ──────────────────────────────────────────────────────────────
# The experiment and search_space definitions are completely reused.
# Swapping optimizers in Hyperactive is a one-line change — this is the
# unified interface the library is built around.