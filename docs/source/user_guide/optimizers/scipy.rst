.. _user_guide_optimizers_scipy:

=============
Scipy Backend
=============

Hyperactive provides wrappers for scipy.optimize algorithms, offering well-tested
implementations for continuous parameter optimization. Scipy optimizers support
only continuous parameter spaces defined as tuples.


Available Optimizers
--------------------

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_imports]
   :end-before: # [end:scipy_imports]


Example: ScipyDifferentialEvolution
-----------------------------------

A robust global optimizer using differential evolution. Handles multi-modal
objective functions well:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_differential_evolution]
   :end-before: # [end:scipy_differential_evolution]


Example: ScipyDualAnnealing
---------------------------

Combines classical simulated annealing with local search. Effective for
problems with many local minima:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_dual_annealing]
   :end-before: # [end:scipy_dual_annealing]


Example: ScipyBasinhopping
--------------------------

Global optimization combining random perturbations with local refinement. Good for
finding global minima in multimodal landscapes:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_basinhopping]
   :end-before: # [end:scipy_basinhopping]


Example: ScipySHGO
------------------

Simplicial Homology Global Optimization. Finds multiple local minima and is
effective for low to moderate dimensional problems:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_shgo]
   :end-before: # [end:scipy_shgo]


Example: ScipyDirect
--------------------

Deterministic global optimizer using the DIRECT (DIviding RECTangles) algorithm.
Requires no random seed and is effective for Lipschitz-continuous functions:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_direct]
   :end-before: # [end:scipy_direct]


Example: ScipyNelderMead
------------------------

A simplex-based local optimizer. Fast convergence for smooth objective functions:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_nelder_mead]
   :end-before: # [end:scipy_nelder_mead]


Example: ScipyPowell
--------------------

Powell's conjugate direction method. A fast local optimizer that can outperform
Nelder-Mead in some cases:

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:scipy_powell]
   :end-before: # [end:scipy_powell]


When to Use Scipy Backend
-------------------------

The Scipy backend is useful when:

- Your parameter space is purely continuous (no categorical or discrete values)
- You want well-tested, production-grade optimization algorithms
- You need specific scipy algorithms not available in other backends
- You prefer scipy's implementation of differential evolution or simulated annealing

Choose ``ScipyDifferentialEvolution`` for robust global optimization.
Choose ``ScipyDualAnnealing`` for problems with many local minima.
Choose ``ScipyBasinhopping`` for global optimization with local refinement.
Choose ``ScipyNelderMead`` or ``ScipyPowell`` for fast local optimization.
