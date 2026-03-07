.. _history:

=======
History
=======

This page documents the history and evolution of Hyperactive.


Project History
---------------

Hyperactive was created in 2018 by Simon Blanke to address the need for a flexible,
unified interface for hyperparameter optimization in machine learning workflows.


Timeline
^^^^^^^^

**2018 - Project Creation**
    Hyperactive was first released as an open-source project, providing a collection
    of gradient-free optimization algorithms accessible through a simple Python API.

**2019 - Growing Adoption**
    The project gained traction in the machine learning community, with users
    appreciating its straightforward interface and variety of optimization algorithms.

**2020-2021 - Ecosystem Expansion**
    Related projects were developed to complement Hyperactive:

    - **Gradient-Free-Optimizers**: The optimization backend was extracted into its
      own package, allowing for more modular development.
    - **Search-Data-Collector**: Tools for saving optimization results.
    - **Search-Data-Explorer**: Visualization dashboard for exploring search data.

**2022-2023 - Continued Development**
    Active maintenance continued with bug fixes, new algorithms, and improved
    documentation. The user base continued to grow.

**2024 - Version 5.0 Redesign**
    Major architecture redesign introducing:

    - **Experiment-based architecture**: Clean separation between optimization
      problems (experiments) and optimization algorithms (optimizers).
    - **Enhanced integrations**: Improved support for scikit-learn, sktime, skpro,
      and PyTorch Lightning.
    - **Optuna backend**: Integration with Optuna's optimization algorithms.
    - **Modern Python support**: Support for Python 3.10 through 3.14.

**2024 - GC.OS Sponsorship**
    Hyperactive became a sponsored project of the German Center for Open Source AI
    (GC.OS), ensuring continued development and maintenance.


Version History
---------------

Major Versions
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Version
     - Highlights
   * - v5.0
     - Experiment-based architecture, Optuna integration, modern Python support
   * - v4.x
     - Improved API stability, additional optimizers
   * - v3.x
     - Search data collection features, expanded algorithm library
   * - v2.x
     - Multi-processing support, warm starting
   * - v1.x
     - Initial public release with core optimization algorithms


Breaking Changes
^^^^^^^^^^^^^^^^

Major version updates (e.g., v4 → v5) may include breaking API changes.
If you're upgrading from an older version:

1. Check the `GitHub releases <https://github.com/SimonBlanke/Hyperactive/releases>`_
   for migration guides.
2. Update your code to use the new API patterns.
3. Alternatively, pin your version to continue using the old API.

.. code-block:: bash

    # Upgrade to latest
    pip install hyperactive --upgrade

    # Or pin to specific version
    pip install hyperactive==4.x.x


Legacy Documentation
^^^^^^^^^^^^^^^^^^^^

Documentation for Hyperactive v4 is still available at the legacy documentation site:

`Legacy Documentation (v4) <https://simonblanke.github.io/hyperactive-documentation/5.0/>`_

This may be useful if you:

- Are maintaining projects that use Hyperactive v4
- Need to reference the previous API design
- Want to compare the old and new approaches


Future Roadmap
--------------

Hyperactive continues to evolve. Planned improvements include:

- Additional optimization algorithms
- Enhanced visualization tools
- Improved distributed computing support
- More framework integrations
- Performance optimizations

For the latest roadmap, see the
`GitHub Issues <https://github.com/SimonBlanke/Hyperactive/issues>`_ and
`Discussions <https://github.com/SimonBlanke/Hyperactive/discussions>`_.
