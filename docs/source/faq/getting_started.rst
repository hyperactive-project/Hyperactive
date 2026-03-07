.. _faq_getting_started:

===============
Getting Started
===============

Which optimizer should I use?
-----------------------------

For most problems, start with one of these recommendations:

**Small search spaces (<100 combinations)**
   Use :class:`~hyperactive.opt.gfo.GridSearch` to exhaustively evaluate all options.

**General-purpose optimization**
   :class:`~hyperactive.opt.gfo.BayesianOptimizer` works well for expensive
   objective functions where you want to minimize evaluations.

**Fast, simple problems**
   :class:`~hyperactive.opt.gfo.HillClimbing` or
   :class:`~hyperactive.opt.gfo.RandomSearch` are good starting points.

**High-dimensional spaces**
   Population-based methods like :class:`~hyperactive.opt.gfo.ParticleSwarmOptimizer`
   or :class:`~hyperactive.opt.gfo.EvolutionStrategyOptimizer` handle many
   parameters well.

See :ref:`user_guide_optimizers` for detailed guidance on choosing optimizers.


How many iterations do I need?
------------------------------

This depends on your search space size and objective function:

- **Rule of thumb**: Start with ``n_iter = 10 * number_of_parameters``
- **Expensive functions**: Use fewer iterations with Bayesian optimization
- **Fast functions**: Use more iterations with simpler optimizers

You can monitor progress and stop early if the score plateaus.


Does Hyperactive minimize or maximize?
--------------------------------------

**Hyperactive maximizes** the objective function. If you want to minimize,
return the negative of your metric:

.. code-block:: python

    def objective(params):
        error = compute_error(params)
        return -error  # Negate to minimize
