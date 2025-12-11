.. _user_guide_integrations:

======================
Framework Integrations
======================

Hyperactive provides seamless integrations with popular machine learning frameworks.
These integrations offer drop-in replacements for tools like ``GridSearchCV``,
making it easy to use any Hyperactive optimizer with your existing code.


Scikit-Learn Integration
------------------------

The ``OptCV`` class provides a scikit-learn compatible interface for hyperparameter
tuning. It works like ``GridSearchCV`` but supports any Hyperactive optimizer.

Basic Usage
^^^^^^^^^^^

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:optcv_basic]
   :end-before: # [end:optcv_basic]


Using Different Optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Any Hyperactive optimizer works with ``OptCV``:

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:different_optimizers]
   :end-before: # [end:different_optimizers]


Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

``OptCV`` works with sklearn pipelines:

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:pipeline_integration]
   :end-before: # [end:pipeline_integration]


Time Series with Sktime
-----------------------

Hyperactive integrates with ``sktime`` for time series forecasting optimization.

.. note::

   Requires ``pip install hyperactive[sktime-integration]``


Forecasting Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``ForecastingOptCV`` to tune forecasters:

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:forecasting_optcv]
   :end-before: # [end:forecasting_optcv]


Time Series Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``TSCOptCV`` for time series classification:

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:tsc_optcv]
   :end-before: # [end:tsc_optcv]


Probabilistic Prediction with Skpro
-----------------------------------

For probabilistic regression with ``skpro``:

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:skpro_experiment]
   :end-before: # [end:skpro_experiment]


PyTorch Lightning Integration
-----------------------------

For deep learning hyperparameter optimization with PyTorch Lightning:

.. note::

   Requires ``pip install hyperactive[all_extras]`` or ``pip install lightning``

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:pytorch_lightning]
   :end-before: # [end:pytorch_lightning]


Choosing the Right Integration
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Framework
     - Integration Class
     - Use Case
   * - scikit-learn
     - ``OptCV``
     - Classification, regression, pipelines
   * - sktime
     - ``ForecastingOptCV``
     - Time series forecasting
   * - sktime
     - ``TSCOptCV``
     - Time series classification
   * - skpro
     - ``SkproProbaRegExperiment``
     - Probabilistic regression
   * - PyTorch Lightning
     - ``TorchExperiment``
     - Deep learning models


Tips for Using Integrations
---------------------------

1. **Match the interface**: Use ``OptCV`` when you want sklearn-compatible behavior
   (fit/predict). Use experiment classes when you want more control.

2. **Consider evaluation cost**: Deep learning experiments are expensive.
   Use efficient optimizers like ``BayesianOptimizer`` with fewer iterations.

3. **Use appropriate CV strategies**: Match your cross-validation to your problem
   (e.g., ``TimeSeriesSplit`` for time series, stratified splits for imbalanced data).

4. **Start simple**: Begin with ``GridSearch`` or ``RandomSearch`` to establish
   baselines before using more sophisticated optimizers.
