.. _user_guide_optimizers_population_based:

==================
Population Methods
==================

Population-based optimizers maintain multiple candidate solutions that evolve together.
They are excellent for complex, multi-modal optimization landscapes.

These methods are inspired by natural processes like evolution and swarm behavior,
using populations of solutions that interact and improve over generations.


Particle Swarm Optimization
---------------------------

Particles "fly" through the search space, influenced by their own best position
and the swarm's best position.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:particle_swarm]
   :end-before: # [end:particle_swarm]


Genetic Algorithm
-----------------

Evolves a population using selection, crossover, and mutation inspired by biology.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:genetic_algorithm]
   :end-before: # [end:genetic_algorithm]


Evolution Strategy
------------------

Similar to genetic algorithms but focused on real-valued optimization.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:evolution_strategy]
   :end-before: # [end:evolution_strategy]


Differential Evolution
----------------------

Uses vector differences to guide mutation. Excellent for continuous optimization.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:differential_evolution]
   :end-before: # [end:differential_evolution]


Parallel Tempering
------------------

Runs multiple chains at different "temperatures" and exchanges information between them.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:parallel_tempering]
   :end-before: # [end:parallel_tempering]


Spiral Optimization
-------------------

Particles spiral toward the best solution found so far.

.. literalinclude:: ../../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:spiral_optimization]
   :end-before: # [end:spiral_optimization]


When to Use Population Methods
------------------------------

Population-based optimizers are best suited for:

- **Multi-modal landscapes**: Problems with many local optima
- **Complex interactions**: When parameters interact in non-obvious ways
- **Robustness**: When you need reliable solutions across different runs
- **Exploration**: When thorough coverage of the search space is important

Consider using model-based methods if:

- Evaluations are very expensive (population methods need many evaluations)
- You want to leverage information from previous evaluations more efficiently
