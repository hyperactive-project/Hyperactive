.. _examples_scipy_backend:

=============
Scipy Backend
=============

Hyperactive provides wrappers for scipy.optimize algorithms, enabling
well-tested, production-grade optimization for continuous parameter spaces.

.. note::

   Scipy must be installed separately:

   .. code-block:: bash

       pip install scipy
       # or
       pip install hyperactive[all_extras]


Available Optimizers
--------------------

The Scipy backend provides 7 optimizers divided into global and local methods.

**Global Optimizers** (5 algorithms):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Optimizer
     - Description
   * - ``ScipyDifferentialEvolution``
     - Population-based global optimizer. Robust for multi-modal landscapes.
   * - ``ScipyDualAnnealing``
     - Combines classical simulated annealing with local search.
   * - ``ScipyBasinhopping``
     - Random perturbations with local minimization. Good for finding global minima.
   * - ``ScipySHGO``
     - Simplicial Homology Global Optimization. Finds multiple local minima.
   * - ``ScipyDirect``
     - Deterministic DIRECT algorithm. No random seed required.

**Local Optimizers** (2 algorithms):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Optimizer
     - Description
   * - ``ScipyNelderMead``
     - Simplex-based optimizer. Fast for smooth functions.
   * - ``ScipyPowell``
     - Conjugate direction method. Often faster than Nelder-Mead.


Quick Example
-------------

Scipy optimizers require continuous parameter spaces defined as tuples:

.. code-block:: python

    from hyperactive.opt.scipy import ScipyDifferentialEvolution

    # Define a continuous search space (tuples, not arrays)
    param_space = {
        "x": (-5.0, 5.0),
        "y": (-5.0, 5.0),
    }

    def objective(params):
        x, y = params["x"], params["y"]
        return -(x**2 + y**2)  # Maximize (minimize negative)

    optimizer = ScipyDifferentialEvolution(
        param_space=param_space,
        n_iter=100,
        experiment=objective,
        random_state=42,
    )

    best_params = optimizer.solve()
    print(f"Best parameters: {best_params}")


When to Use Scipy Backend
-------------------------

The Scipy backend is useful when:

- **Continuous parameters only**: Your search space has no categorical or discrete values
- **Production-grade algorithms**: You need well-tested, reliable implementations
- **Specific scipy features**: You want scipy's differential evolution or simulated annealing
- **Deterministic optimization**: Use ``ScipyDirect`` for reproducible results without random seeds


See Also
--------

- :ref:`user_guide_optimizers_scipy` - Detailed guide with all optimizer examples
- :ref:`optimizers_scipy_ref` - API reference for all Scipy optimizers
