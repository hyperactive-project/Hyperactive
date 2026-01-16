<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./docs/images/hyperactive_logo_ink_dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="./docs/images/hyperactive_logo_ink.svg">
      <img src="./docs/images/hyperactive_logo_ink.svg" width="400" alt="Hyperactive Logo">
    </picture>
  </a>
</p>

---

<h3 align="center">
A unified interface for optimization algorithms and experiments in Python.
</h3>

<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive/actions"><img src="https://img.shields.io/github/actions/workflow/status/SimonBlanke/Hyperactive/test.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/SimonBlanke/Hyperactive"><img src="https://img.shields.io/codecov/c/github/SimonBlanke/Hyperactive?style=flat-square" alt="Coverage"></a>
</p>

<table align="center">
  <tr>
    <td align="right"><b>Documentation</b></td>
    <td align="center">&#9656;</td>
    <td>
      <a href="https://hyperactive.readthedocs.io/en/latest/">Homepage</a> &#183;
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide.html">User Guide</a> &#183;
      <a href="https://hyperactive.readthedocs.io/en/latest/api_reference.html">API Reference</a> &#183;
      <a href="https://hyperactive.readthedocs.io/en/latest/examples.html">Examples</a>
    </td>
  </tr>
  <tr>
    <td align="right"><b>On this page</b></td>
    <td align="center">&#9656;</td>
    <td>
      <a href="#key-features">Features</a> &#183;
      <a href="#examples">Examples</a> &#183;
      <a href="#core-concepts">Concepts</a> &#183;
      <a href="#citation">Citation</a>
    </td>
  </tr>
</table>

<br>

---

<a href="https://github.com/SimonBlanke/Hyperactive">
  <img src="./docs/images/bayes_ackley.gif" width="240" align="right" alt="Bayesian Optimization on Ackley Function">
</a>

**Hyperactive** provides 31 optimization algorithms across 3 backends (GFO, Optuna, scikit-learn), accessible through a unified experiment-based interface. The library separates optimization problems from algorithms, enabling you to swap optimizers without changing your experiment code.

Designed for hyperparameter tuning, model selection, and black-box optimization. Native integrations with scikit-learn, sktime, skpro, and PyTorch allow tuning ML models with minimal setup. Define your objective, specify a search space, and run.

<p>
  <a href="https://www.linkedin.com/company/german-center-for-open-source-ai"><img src="https://img.shields.io/badge/LinkedIn-Follow-0A66C2?style=flat-square&logo=linkedin" alt="LinkedIn"></a>
  <a href="https://discord.gg/7uKdHfdcJG"><img src="https://img.shields.io/badge/Discord-Chat-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/sponsors/SimonBlanke"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=flat-square&logo=githubsponsors&logoColor=white" alt="Sponsor"></a>
</p>

---

## Installation

```bash
pip install hyperactive
```

<p>
  <a href="https://pypi.org/project/hyperactive/"><img src="https://img.shields.io/pypi/v/hyperactive?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/hyperactive/"><img src="https://img.shields.io/pypi/pyversions/hyperactive?style=flat-square" alt="Python"></a>
</p>

<details>
<summary>Optional dependencies</summary>

```bash
pip install hyperactive[sklearn-integration]  # scikit-learn integration
pip install hyperactive[sktime-integration]   # sktime/skpro integration
pip install hyperactive[all_extras]           # Everything including Optuna
```

</details>

---

## Key Features

<table>
  <tr>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide/optimizers/index.html"><b>31 Optimization Algorithms</b></a><br>
      <sub>Local, global, population-based, and model-based methods across 3 backends (GFO, Optuna, sklearn).</sub>
    </td>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide/experiments.html"><b>Experiment Abstraction</b></a><br>
      <sub>Clean separation between what to optimize (experiments) and how to optimize (algorithms).</sub>
    </td>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide/search_spaces.html"><b>Flexible Search Spaces</b></a><br>
      <sub>Discrete, continuous, and mixed parameter types. Define spaces with NumPy arrays or lists.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide/integrations.html"><b>ML Framework Integrations</b></a><br>
      <sub>Native support for scikit-learn, sktime, skpro, and PyTorch with minimal code changes.</sub>
    </td>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/user_guide/optimizers/optuna.html"><b>Multiple Backends</b></a><br>
      <sub>GFO algorithms, Optuna samplers, and sklearn search methods through one unified API.</sub>
    </td>
    <td width="33%">
      <a href="https://hyperactive.readthedocs.io/en/latest/api_reference.html"><b>Production Ready</b></a><br>
      <sub>5+ years of development, comprehensive test coverage, and active maintenance since 2019.</sub>
    </td>
  </tr>
</table>

---

## Quick Start

```python
import numpy as np
from hyperactive.opt.gfo import HillClimbing

# Define objective function (maximize)
def objective(params):
    x, y = params["x"], params["y"]
    return -(x**2 + y**2)  # Negative paraboloid, optimum at (0, 0)

# Define search space
search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

# Run optimization
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=objective,
)
best_params = optimizer.solve()

print(f"Best params: {best_params}")
```

**Output:**
```
Best params: {'x': 0.0, 'y': 0.0}
```

---

## Core Concepts

```
                    EXPERIMENT-BASED ARCHITECTURE

    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Optimizer  │───>│    Search    │───>│  Experiment  │
    │  (Algorithm) │    │    Space     │    │  (Objective) │
    └──────────────┘    └──────────────┘    └──────────────┘
           │                   │                    │
           │                   │                    │
           v                   v                    v
    ┌────────────────────────────────────────────────────┐
    │                    Best Parameters                  │
    │           optimizer.solve() -> best_params          │
    └────────────────────────────────────────────────────┘
```

**Optimizer**: Implements the search strategy (Hill Climbing, Bayesian, Particle Swarm, etc.).

**Search Space**: Defines valid parameter combinations as NumPy arrays or lists.

**Experiment**: Your objective function or a built-in experiment (SklearnCvExperiment, etc.).

**Best Parameters**: The optimizer returns the parameters that maximize the objective.

---

## Examples

<details open>
<summary><b>Scikit-learn Hyperparameter Tuning</b></summary>

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import HillClimbing

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define search space and optimizer
search_space = {"kernel": ["linear", "rbf"], "C": [1, 10, 100]}
optimizer = HillClimbing(search_space=search_space, n_iter=20)

# Create tuned estimator
tuned_svc = OptCV(SVC(), optimizer)
tuned_svc.fit(X_train, y_train)

print(f"Best params: {tuned_svc.best_params_}")
print(f"Test accuracy: {tuned_svc.score(X_test, y_test):.3f}")
```

</details>

<br>

<details>
<summary><b>Bayesian Optimization</b></summary>

```python
import numpy as np
from hyperactive.opt.gfo import BayesianOptimizer

def ackley(params):
    x, y = params["x"], params["y"]
    return -(
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e + 20
    )

search_space = {
    "x": np.arange(-5, 5, 0.01),
    "y": np.arange(-5, 5, 0.01),
}

optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=50,
    experiment=ackley,
)
best_params = optimizer.solve()
```

</details>

<br>

<details>
<summary><b>Particle Swarm Optimization</b></summary>

```python
import numpy as np
from hyperactive.opt.gfo import ParticleSwarmOptimizer

def rastrigin(params):
    A = 10
    values = [params[f"x{i}"] for i in range(5)]
    return -sum(v**2 - A * np.cos(2 * np.pi * v) + A for v in values)

search_space = {f"x{i}": np.arange(-5.12, 5.12, 0.1) for i in range(5)}

optimizer = ParticleSwarmOptimizer(
    search_space=search_space,
    n_iter=500,
    experiment=rastrigin,
    population_size=20,
)
best_params = optimizer.solve()
```

</details>

<br>

<details>
<summary><b>Experiment Abstraction with SklearnCvExperiment</b></summary>

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import HillClimbing

X, y = load_iris(return_X_y=True)

# Create reusable experiment
sklearn_exp = SklearnCvExperiment(
    estimator=SVC(),
    scoring=accuracy_score,
    cv=KFold(n_splits=3, shuffle=True),
    X=X,
    y=y,
)

search_space = {
    "C": np.logspace(-2, 2, num=10),
    "kernel": ["linear", "rbf"],
}

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=sklearn_exp,
)
best_params = optimizer.solve()
```

</details>

<br>

<details>
<summary><b>Optuna Backend (TPE)</b></summary>

```python
import numpy as np
from hyperactive.opt.optuna import TPEOptimizer

def objective(params):
    x, y = params["x"], params["y"]
    return -(x**2 + y**2)

search_space = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}

optimizer = TPEOptimizer(
    search_space=search_space,
    n_iter=100,
    experiment=objective,
)
best_params = optimizer.solve()
```

</details>

<br>

<details>
<summary><b>Time Series Forecasting with sktime</b></summary>

```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.datasets import load_airline

from hyperactive.integrations.sktime import ForecastingOptCV
from hyperactive.opt.gfo import RandomSearch

y = load_airline()

search_space = {
    "strategy": ["last", "mean", "drift"],
    "sp": [1, 12],
}

optimizer = RandomSearch(search_space=search_space, n_iter=10)
tuned_forecaster = ForecastingOptCV(NaiveForecaster(), optimizer)
tuned_forecaster.fit(y)

print(f"Best params: {tuned_forecaster.best_params_}")
```

</details>

<br>

<details>
<summary><b>PyTorch Hyperparameter Tuning</b></summary>

```python
import numpy as np
from hyperactive.opt.gfo import BayesianOptimizer

def train_model(params):
    # Your PyTorch model training here
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    hidden_size = params["hidden_size"]

    # ... training code ...
    # return validation_accuracy
    pass

search_space = {
    "learning_rate": np.logspace(-5, -1, 20),
    "batch_size": [16, 32, 64, 128],
    "hidden_size": [64, 128, 256, 512],
}

optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=30,
    experiment=train_model,
)
best_params = optimizer.solve()
```

</details>

---

## Ecosystem

This library is part of a suite of optimization and machine learning tools. For updates on these packages, [follow on GitHub](https://github.com/SimonBlanke).

| Package | Description |
|---------|-------------|
| [Hyperactive](https://github.com/SimonBlanke/Hyperactive) | Hyperparameter optimization framework with experiment abstraction and ML integrations |
| [Gradient-Free-Optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers) | Core optimization algorithms for black-box function optimization |
| [Surfaces](https://github.com/SimonBlanke/Surfaces) | Test functions and benchmark surfaces for optimization algorithm evaluation |

---

## Documentation

| Resource | Description |
|----------|-------------|
| [User Guide](https://hyperactive.readthedocs.io/en/latest/user_guide.html) | Comprehensive tutorials and explanations |
| [API Reference](https://hyperactive.readthedocs.io/en/latest/api_reference.html) | Complete API documentation |
| [Examples](https://hyperactive.readthedocs.io/en/latest/examples.html) | Jupyter notebooks with use cases |
| [FAQ](https://hyperactive.readthedocs.io/en/latest/faq.html) | Common questions and troubleshooting |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

- **Bug reports**: [GitHub Issues](https://github.com/SimonBlanke/Hyperactive/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/SimonBlanke/Hyperactive/discussions)
- **Questions**: [Discord](https://discord.gg/7uKdHfdcJG)

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hyperactive2019,
  author = {Simon Blanke},
  title = {Hyperactive: A hyperparameter optimization and meta-learning toolbox},
  year = {2019},
  url = {https://github.com/SimonBlanke/Hyperactive},
}
```

---

## License

[MIT License](./LICENSE) - Free for commercial and academic use.
