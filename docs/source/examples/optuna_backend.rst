.. _examples_optuna_backend:

==============
Optuna Backend
==============

Hyperactive provides wrappers for Optuna's optimization algorithms, allowing
you to use Optuna's powerful samplers with Hyperactive's interface.

.. note::

   Optuna must be installed separately:

   .. code-block:: bash

       pip install optuna
       # or
       pip install hyperactive[all_extras]


Sampler Examples
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sampler
     - Example
   * - TPE (Tree-Parzen Estimator)
     - `tpe_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/tpe_sampler_example.py>`_
   * - CMA-ES
     - `cmaes_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/cmaes_sampler_example.py>`_
   * - Gaussian Process
     - `gp_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/gp_sampler_example.py>`_
   * - NSGA-II
     - `nsga_ii_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/nsga_ii_sampler_example.py>`_
   * - NSGA-III
     - `nsga_iii_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/nsga_iii_sampler_example.py>`_
   * - QMC (Quasi-Monte Carlo)
     - `qmc_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/qmc_sampler_example.py>`_
   * - Random
     - `random_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/random_sampler_example.py>`_
   * - Grid
     - `grid_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/grid_sampler_example.py>`_


When to Use Optuna Backend
--------------------------

The Optuna backend is useful when you need:

- **Multi-objective optimization** (NSGA-II, NSGA-III)
- **Advanced sampling strategies** like CMA-ES or QMC
- **Optuna's pruning capabilities** for early stopping
- **Compatibility** with existing Optuna workflows
