.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to Hyperactive! This guide will help
you get started with development and submit your contributions.


How to Contribute
-----------------

Contribution Workflow
^^^^^^^^^^^^^^^^^^^^^

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** with tests
5. **Run the test suite** to ensure everything works
6. **Submit a pull request** for review


Types of Contributions
^^^^^^^^^^^^^^^^^^^^^^

We welcome many types of contributions:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add new optimizers, experiments, or integrations
- **Documentation**: Improve guides, examples, and API docs
- **Tests**: Increase test coverage
- **Performance**: Optimize code for speed or memory


Development Setup
-----------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.10 or higher
- Git
- pip

Setting Up Your Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fork and clone the repository:

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/Hyperactive.git
       cd Hyperactive

2. Create a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode with test dependencies:

   .. code-block:: bash

       pip install -e ".[test,docs]"

4. Verify the installation:

   .. code-block:: bash

       python -c "import hyperactive; print(hyperactive.__version__)"


Running Tests
^^^^^^^^^^^^^

Run the test suite to ensure everything works:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage report
    pytest --cov=hyperactive

    # Run specific test file
    pytest tests/test_specific.py

    # Run tests matching a pattern
    pytest -k "test_hill_climbing"


Code Style
----------

Formatting
^^^^^^^^^^

Hyperactive uses `Black <https://black.readthedocs.io/>`_ for code formatting
and `Ruff <https://docs.astral.sh/ruff/>`_ for linting:

.. code-block:: bash

    # Format code
    black src/hyperactive tests

    # Check linting
    ruff check src/hyperactive tests

    # Auto-fix linting issues
    ruff check --fix src/hyperactive tests


Docstrings
^^^^^^^^^^

Use NumPy-style docstrings for all public functions and classes:

.. code-block:: python

    def my_function(param1, param2):
        """Short description of the function.

        Longer description if needed.

        Parameters
        ----------
        param1 : type
            Description of param1.
        param2 : type
            Description of param2.

        Returns
        -------
        type
            Description of return value.

        Examples
        --------
        >>> my_function(1, 2)
        3
        """
        return param1 + param2


Type Hints
^^^^^^^^^^

Add type hints to function signatures:

.. code-block:: python

    def optimize(
        self,
        search_space: dict,
        n_iter: int,
        experiment: Callable,
    ) -> dict:
        ...


Submitting Changes
------------------

Creating a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^

1. **Create a branch** for your changes:

   .. code-block:: bash

       git checkout -b feature/my-new-feature

2. **Make your changes** and commit:

   .. code-block:: bash

       git add .
       git commit -m "Add my new feature"

3. **Push to your fork**:

   .. code-block:: bash

       git push origin feature/my-new-feature

4. **Open a pull request** on GitHub from your branch to the main repository.


Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

- **Clear title**: Describe what the PR does
- **Description**: Explain the changes and motivation
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs if needed
- **Small scope**: Keep PRs focused on one thing


Commit Messages
^^^^^^^^^^^^^^^

Write clear, descriptive commit messages:

.. code-block:: text

    Add Bayesian optimizer warm start support

    - Add warm_start parameter to BayesianOptimizer
    - Update documentation with usage examples
    - Add tests for warm start functionality


Review Process
--------------

What to Expect
^^^^^^^^^^^^^^

1. **Automated checks**: CI will run tests and linting
2. **Code review**: Maintainers will review your code
3. **Feedback**: You may be asked to make changes
4. **Merge**: Once approved, your PR will be merged


Response Time
^^^^^^^^^^^^^

Maintainers are volunteers, so response times may vary. We aim to:

- Acknowledge PRs within a few days
- Provide initial review within a week
- Merge approved PRs promptly


Getting Help
------------

If you need help:

- Check existing `issues <https://github.com/SimonBlanke/Hyperactive/issues>`_
  and `discussions <https://github.com/SimonBlanke/Hyperactive/discussions>`_
- Ask on `Discord <https://discord.gg/7uKdHfdcJG>`_
- Tag @SimonBlanke in your PR for attention

Thank you for contributing to Hyperactive!
