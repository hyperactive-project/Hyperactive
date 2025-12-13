
## Welcome to hyperactive

<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive"><img src="./docs/images/logo.png" width="300" align="right"></a>
</p>

**A unified interface for optimization algorithms and problems.**

Hyperactive implements a collection of optimization algorithms, accessible through a unified experiment-based
interface that separates optimization problems from algorithms. The library provides native implementations of algorithms from the Gradient-Free-Optimizers
package alongside direct interfaces to Optuna and scikit-learn optimizers, supporting discrete, continuous, and mixed parameter spaces.


<br>

---

| | [Overview](https://github.com/SimonBlanke/Hyperactive#overview) • [Installation](https://github.com/SimonBlanke/Hyperactive#installation) • [Tutorial](https://nbviewer.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/hyperactive_tutorial.ipynb) • [API reference](https://hyperactive.readthedocs.io/en/latest/#) • [Citation](https://github.com/SimonBlanke/Hyperactive#citing-hyperactive) |
|---|---|
| **Open&#160;Source** | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Community** | [![Discord](https://img.shields.io/static/v1?logo=discord&label=Discord&message=chat&color=lightgreen)](https://discord.gg/7uKdHfdcJG) [![LinkedIn](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/german-center-for-open-source-ai)  |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/SimonBlanke/hyperactive/test.yml?logo=github)](https://github.com/SimonBlanke/hyperactive/actions/workflows/test.yml) [![readthedocs](https://img.shields.io/readthedocs/hyperactive?logo=readthedocs)](https://www.hyperactive.net/en/latest/?badge=latest)
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/hyperactive?color=orange)](https://pypi.org/project/hyperactive/) [![!python-versions](https://img.shields.io/pypi/pyversions/hyperactive)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |

## Installation

```console
pip install hyperactive
```

## Key Concepts

### Experiment-Based Architecture

Hyperactive v5 introduces a clean separation between optimization algorithms and optimization problems through the **experiment abstraction**:

- **Experiments** define *what* to optimize (the objective function and evaluation logic)
- **Optimizers** define *how* to optimize (the search strategy and algorithm)

This design allows you to:
- Mix and match any optimizer with any experiment type
- Create reusable experiment definitions for common ML tasks
- Easily switch between different optimization strategies
- Build complex optimization workflows with consistent interfaces

**Built-in experiments include:**
- `SklearnCvExperiment` - Cross-validation for sklearn estimators
- `SktimeForecastingExperiment` - Time series forecasting optimization
- Custom function experiments (pass any callable as experiment)


## Quickstart

### Maximizing a custom function

```python
import numpy as np

# function to be maximized
def problem(params):
    x = params["x"]
    y = params["y"]

    return -(x**2 + y**2)

# discrete search space: dict of iterable, scikit-learn like grid space
# (valid search space types depends on optimizer)
search_space = {
    "x": np.arange(-1, 1, 0.01),
    "y": np.arange(-1, 2, 0.1),
}

from hyperactive.opt.gfo import HillClimbing

hillclimbing = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=problem,
)

# running the hill climbing search:
best_params = hillclimbing.solve()
```

### experiment abstraction - example: scikit-learn CV experiment

"experiment" abstraction = parametrized optimization problem

`hyperactive` provides a number of common experiments, e.g.,
`scikit-learn` cross-validation experiments:

```python
import numpy as np
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

X, y = load_iris(return_X_y=True)

# create experiment
sklearn_exp = SklearnCvExperiment(
    estimator=SVC(),
    scoring=accuracy_score,
    cv=KFold(n_splits=3, shuffle=True),
    X=X,
    y=y,
)

# experiments can be evaluated via "score"
params = {"C": 1.0, "kernel": "linear"}
score, add_info = sklearn_exp.score(params)

# they can be used in optimizers like above
from hyperactive.opt.gfo import HillClimbing

search_space = {
    "C": np.logspace(-2, 2, num=10),
    "kernel": ["linear", "rbf"],
}

hillclimbing = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=sklearn_exp,
)

best_params = hillclimbing.solve()
```

### full ML toolbox integration - example: scikit-learn

Any `hyperactive` optimizer can be combined with the ML toolbox integrations!

`OptCV` for tuning `scikit-learn` estimators with any `hyperactive` optimizer:

```python
# 1. defining the tuned estimator:
from sklearn.svm import SVC
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import HillClimbing

search_space = {"kernel": ["linear", "rbf"], "C": [1, 10]}
optimizer = HillClimbing(search_space=search_space, n_iter=20)
tuned_svc = OptCV(SVC(), optimizer)

# 2. fitting the tuned estimator:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tuned_svc.fit(X_train, y_train)

y_pred = tuned_svc.predict(X_test)

# 3. obtaining best parameters and best estimator
best_params = tuned_svc.best_params_
best_estimator = tuned_svc.best_estimator_
```






<br>

## Citing Hyperactive

    @Misc{hyperactive2021,
      author =   {{Simon Blanke}},
      title =    {{Hyperactive}: An optimization and data collection toolbox for convenient and fast prototyping of computationally expensive models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


