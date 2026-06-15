# Changelog

All notable changes to Hyperactive are documented in this file.

## [v5.0.4] - 2026-03-06

### Documentation
- Add missing torch entries in README and documentation (#239)
- Update landing page and styling
- Change formatting and key features in README
- Rework entire README with mermaid diagram and updated links
- Fix links, add overview diagram hover effect
- Hide toc parents

### Bug Fixes
- Fix `_score_params` and add tests; fix errors masked by `try... except` block (#237)
- Add `warnings` to exception and fix bug
- Override `_predict_proba`

### Maintenance
- Fix formatting
- Remove unused dependency
- Remove workflow
- Update CONTRIBUTING.md
- Update PULL_REQUEST_TEMPLATE.md
- Change `n_iter` in examples


## [v5.0.3] - 2025-12-15

### Documentation
- Preload font, add animation of right sidebar
- Add explanations, change title styling, add links
- Rework user-guide, add diagrams
- Change API reference optimizer pages
- Add light/dark logo, fix logo font/text
- Cleanup README, update links
- Add link to legacy docs
- Add nav-bar logo, change title and sidebar
- Rework landing page: Quick Install, Examples, Integrations, Optimization backends
- Add multiple pages and small corrections
- Separate examples pages
- Add Python examples via `literalinclude` with tests
- Dynamically get supported Python versions, update CSS
- Add static content and readthedocs skeleton

### Maintenance
- Add keywords and classifiers to `pyproject.toml`
- Ensure tests are not skipped
- Remove custom Docker container from CI, add Dockerfile
- Move doctests, add dependency, fix doctest
- Add `--no-cache-dir` flag, run targeted tests first
- Add auto-pr-label and draft-changelog workflows
- Add PR template
- Python 3.14 compatibility and Python 3.9 end-of-life (#202)

### Enhancements
- Add PyTorch Lightning integration (#203)
- `skpro` integration (#200)

### Bug Fixes
- Fix cmaes optimizer tests

### Dependencies
- Bump pytest from 9.0.1 to 9.0.2 (#218)
- Bump pytest from 8.4.2 to 9.0.1 (#206)


## [v5.0.2] - 2025-09-20

### Bug Fixes
- Fix `TSCOptCV` integration for metric function input (#190)


## [v5.0.1] - 2025-09-19

### Enhancements
- Add optuna optional import (#187)

### Documentation
- Minor improvements to README: white background for logo, link table formatting (#186)


## [v5.0.0] - 2025-09-14

### Breaking Changes
- Complete v5 API redesign (#185)
- Change `BaseOptimizer.run` method to `BaseOptimizer.solve` (#159)
- Rename `toy` module to `bench` (#164)
- Experiments: uniform call signature and terminology (#178)
- Refactor callbacks/catch parameter
- Remove `add_search` method

### Enhancements
- `sktime` integration for time series classification (#173)
- `sktime` integration for forecasters (#157)
- `optuna` optimizer interface (#155)
- Add optimization algorithms from GFO (#127)
- Parallelization backends for grid and random search (#150, #162)
- Allowing old function API to be passed as experiment (#152)
- Sign handling in experiments and optimization (#142)
- Quick testing utility `check_estimator` (#130)
- Implement sklearn-approved experiment class
- Add composite optimizer class (enabling parallel via `add` method)

### Bug Fixes
- Fix selection direction, scorer handling, and fit kwargs; resolve sktime doctest (#182)
- Fix `params` ignored in `SktimeForecastingExperiment` (#175)
- Fix docstrings of benchmark toy experiments (#171)
- Fix sklearn v1.7 compatibility (#138)

### Maintenance
- Lint entire repository using `pre-commit` (#156)
- Code quality job (#151)
- Raise `scikit-learn` bound to `<1.8` (#145, #149)
- Cleanup v4 (#165)
- Change `master` to `main` (#166)
- Restore sklearn integration tests (#167)
- Change examples to v5 (#168)

### Dependencies
- Bump pytest from 8.4.0 to 8.4.2 (#129, #137, #181)


## [v4.8.0] - 2024-08-14

### Enhancements
- Add support for numpy v2 and pandas v2
- Add testing for Python 3.12

### Maintenance
- Transfer `setup.py` to `pyproject.toml`
- Change project structure to src-layout
- Read version from pyproject file


## [v4.7.0] - 2024-07-29

### Enhancements
- Add Genetic Algorithm and Differential Evolution optimizers
- Add constrained optimization example to README

### Maintenance
- Add linter
- Remove Python 3.5, 3.6, 3.7 from supported versions
- Update requirement files


## [v4.6.0] - 2023-10-27

### Enhancements
- Add constrained optimization support (constraints parameter in API)
- Add constraint class with support in optimization strategies

### Documentation
- Add examples: constrained optimization, grid search, Lipschitz optimization, stochastic hill climbing, direct algorithm, downhill simplex, Powell's method, pattern search, spiral optimization


## [v4.5.0] - 2023-08-27

### Enhancements
- Add early stopping support in optimization strategies
- Print additional results from the objective function
- Pass `early_stopping` parameter to optimization backend
- Add type hints for basic API

### Bug Fixes
- Log warning if getting max score fails (when all scores are NaN)
- Fix verbosity for progress-bar
- Remove empty lines on no verbosity (#71)


## [v4.4.0] - 2023-03-01

### Enhancements
- Add optimization strategy feature with custom strategies
- Add tqdm progress-bar with description and postfix
- Add SpiralOptimization and LipschitzOptimizer
- Add Direct Algorithm
- Add Ray support for parallelization
- Add `times` parameter to `.search_data()` for eval and iter times
- Add callbacks feature
- Add exception handling (catch parameter)
- Add `pass_through` parameter
- Add SMBO warm start support in optimization strategies (#54)

### Bug Fixes
- Create deepcopy of optimizer (#62)

### Maintenance
- Extend Python versions to 3.11
- Add contributing.md

### Dependencies
- Bump GFO requirement version


## [v4.0.0] - 2021-12-01

Initial v4 release.
