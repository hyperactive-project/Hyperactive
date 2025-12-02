clean: clean-pyc clean-ipynb clean-catboost clean-build clean-test

clean-progress_board:
	find . -name '*.csv~' -exec rm -f {} +
	find . -name '*.lock~' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-ipynb:
	find . -name '*.ipynb_checkpoints' -exec rm -fr {} +

clean-catboost:
	find . -name 'catboost_info' -exec rm -fr {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-test:
	cd tests/; \
		rm -f .coverage; \
		rm -fr htmlcov/

test-search_space:
	cd tests/; \
		i=0; while [ "$$i" -le 100 ]; do \
			i=$$((i + 1));\
			pytest -q test_search_spaces.py; \
	done

test-pytest:
	python -m pytest --durations=10 -x -p  no:warnings tests/; \

test-src:
	python -m pytest --durations=10 --verbose -x -p  no:warnings src/hyperactive/; \

test-timings:
	cd tests/_local_test_timings; \
		pytest *.py -x -p no:warnings

test-local: test-timings

test:  test-src test-pytest test-local


test-examples:
	cd tests; \
		python _test_examples.py

test-extensive: test test-local test-examples

push: test
	git push

release: reinstall test-extensive
	python -m twine upload dist/*

dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y hyperactive
	rm -fr build dist *.egg-info

install-test-requirements:
	python -m pip install .[test]

install-build-requirements:
	python -m pip install .[build]

install-all-extras:
	python -m pip install .[all_extras]

install-no-extras-for-test:
	python -m pip install .[test]

install-all-extras-for-test:
	python -m pip install .[all_extras,test,test_parallel_backends,sktime-integration]

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

# === Linting and Formatting Commands ===

# Run ruff linter to check for code issues
lint:
	ruff check .

# Run ruff linter with auto-fix for fixable issues
lint-fix:
	ruff check --fix .

# Format code using ruff formatter
format:
	ruff format .

# Check formatting without making changes
format-check:
	ruff format --check .

# Sort imports using ruff (isort functionality)
isort:
	ruff check --select I --fix .

# Check import sorting without making changes
isort-check:
	ruff check --select I .

# Run all code quality checks (lint + format check + import check)
check: lint format-check isort-check

# Fix all auto-fixable issues (lint + format + imports)
fix: lint-fix format isort

# === Notebook-specific Commands ===

# Run ruff on Jupyter notebooks
lint-notebooks:
	ruff check --include="*.ipynb" .

# Fix ruff issues in Jupyter notebooks
lint-notebooks-fix:
	ruff check --include="*.ipynb" --fix .

# Format Jupyter notebooks with black via nbqa
format-notebooks:
	pre-commit run nbqa-black --all-files

# Run all notebook checks and fixes
notebooks-fix: lint-notebooks-fix format-notebooks

# === Pre-commit Commands ===

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit:
	pre-commit run

# Update pre-commit hooks to latest versions
pre-commit-update:
	pre-commit autoupdate

# === Combined Quality Commands ===

# Run comprehensive code quality checks
quality-check: check lint-notebooks

# Fix all code quality issues
quality-fix: fix notebooks-fix

# Full quality workflow: install hooks, fix issues, run final check
quality: pre-commit-install quality-fix quality-check
