.. _examples_general:

================
General Examples
================

These examples demonstrate Hyperactive's core functionality with simple,
illustrative use cases.


Running Examples
----------------

All examples are available in the
`examples directory <https://github.com/SimonBlanke/Hyperactive/tree/master/examples>`_
on GitHub. You can run any example directly:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Hyperactive.git
    cd Hyperactive/examples

    # Run an example
    python gfo/hill_climbing_example.py


Custom Function Optimization
----------------------------

The simplest use case: optimizing a mathematical function.

.. literalinclude:: ../_snippets/examples/basic_examples.py
   :language: python
   :start-after: # [start:custom_function]
   :end-before: # [end:custom_function]


Scikit-learn Model Tuning
-------------------------

Hyperparameter optimization for machine learning models.

.. literalinclude:: ../_snippets/examples/basic_examples.py
   :language: python
   :start-after: # [start:sklearn_tuning]
   :end-before: # [end:sklearn_tuning]
