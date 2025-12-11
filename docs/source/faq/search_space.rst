.. _faq_search_space:

======================
Search Space Questions
======================

How do I define a continuous search space?
------------------------------------------

Use NumPy to create arrays of values:

.. code-block:: python

    import numpy as np

    search_space = {
        "learning_rate": np.logspace(-4, -1, 50),  # 0.0001 to 0.1
        "momentum": np.linspace(0.5, 0.99, 50),    # 0.5 to 0.99
    }

Hyperactive samples from these arrays, so finer granularity gives
more precision at the cost of a larger search space.


Can I mix discrete and continuous parameters?
---------------------------------------------

Yes, mix freely:

.. code-block:: python

    search_space = {
        "n_estimators": [10, 50, 100, 200],          # Discrete
        "max_depth": list(range(3, 20)),             # Discrete range
        "learning_rate": np.linspace(0.01, 0.3, 30), # Continuous
        "algorithm": ["SAMME", "SAMME.R"],           # Categorical
    }


How do I include None as a parameter value?
-------------------------------------------

Include ``None`` directly in your list:

.. code-block:: python

    search_space = {
        "max_depth": [None, 3, 5, 10, 20],
    }
