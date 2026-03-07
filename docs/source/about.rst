.. _about:

=====
About
=====

Hyperactive is an optimization and data collection toolbox for convenient and fast
prototyping of computationally expensive models.

.. toctree::
   :maxdepth: 1

   about/team
   about/history
   about/license


About Hyperactive
-----------------

Hyperactive provides a unified interface for hyperparameter optimization using
various gradient-free optimization algorithms. It supports optimization for
scikit-learn, sktime, skpro, and PyTorch Lightning models, as well as custom
objective functions.


Mission
^^^^^^^

Hyperactive aims to make hyperparameter optimization accessible and practical for
machine learning practitioners. By providing a unified API across many optimization
algorithms and ML frameworks, it reduces the barrier to finding optimal model
configurations.


Key Features
^^^^^^^^^^^^

- **20+ Optimization Algorithms**: From simple hill climbing to advanced Bayesian
  optimization, population methods, and Optuna integration.

- **Experiment-Based Architecture**: Clean separation between what to optimize
  (experiments) and how to optimize (algorithms).

- **Framework Integrations**: First-class support for scikit-learn, sktime, skpro,
  and PyTorch Lightning.

- **Flexible Search Spaces**: Discrete, continuous, and mixed parameter spaces
  using familiar NumPy/list syntax.

- **Production Ready**: Battle-tested since 2019 with comprehensive testing and
  active maintenance.


Related Projects
^^^^^^^^^^^^^^^^

Hyperactive is part of a larger ecosystem:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Project
     - Description
   * - `Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
     - Core optimization algorithms used by Hyperactive
   * - `Search-Data-Collector <https://github.com/SimonBlanke/search-data-collector>`_
     - Save search data during optimization to CSV files
   * - `Search-Data-Explorer <https://github.com/SimonBlanke/search-data-explorer>`_
     - Visualize search data with Plotly in a Streamlit dashboard


Sponsorship
^^^^^^^^^^^

Hyperactive is sponsored by the
`German Center for Open Source AI (GC.OS) <https://gc-os-ai.github.io/>`_.

.. image:: https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=for-the-badge&colorA=0eac92&colorB=2077b4
   :target: https://gc-os-ai.github.io/
   :alt: GC.OS Sponsored


Citing Hyperactive
^^^^^^^^^^^^^^^^^^

If you use Hyperactive in your research, please cite it:

.. code-block:: bibtex

    @Misc{hyperactive2021,
      author =   {{Simon Blanke}},
      title =    {{Hyperactive}: An optimization and data collection toolbox
                  for convenient and fast prototyping of computationally
                  expensive models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


Community
^^^^^^^^^

- **GitHub**: `SimonBlanke/Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_
- **Discord**: `Join the community <https://discord.gg/7uKdHfdcJG>`_
- **LinkedIn**: `German Center for Open Source AI <https://www.linkedin.com/company/german-center-for-open-source-ai>`_
