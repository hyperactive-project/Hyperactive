.. _user_guide_integrations:

======================
Framework Integrations
======================

Hyperactive integrates with popular ML frameworks, providing drop-in replacements
for tools like ``GridSearchCV``. Each ML framework has its own conventions for training and evaluation. The integration
classes handle cross-validation setup, scoring metrics, and parameter translation, so
you can use any optimizer with scikit-learn, sktime, skpro, or PyTorch models.

----

Supported Frameworks
--------------------

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: scikit-learn
      :class-card: sd-border-primary
      :link: #scikit-learn-integration
      :link-type: url

      **OptCV**

      Classification, regression, pipelines

   .. grid-item-card:: sktime
      :class-card: sd-border-success
      :link: #time-series-with-sktime
      :link-type: url

      **ForecastingOptCV, TSCOptCV**

      Time series forecasting & classification

   .. grid-item-card:: skpro
      :class-card: sd-border-warning
      :link: #probabilistic-prediction-with-skpro
      :link-type: url

      **SkproProbaRegExperiment**

      Probabilistic regression

   .. grid-item-card:: PyTorch
      :class-card: sd-border-danger
      :link: #pytorch-lightning-integration
      :link-type: url

      **TorchExperiment**

      Deep learning models

----

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Framework
     - Class
     - Use Case
     - Install Extra
   * - scikit-learn
     - ``OptCV``
     - Classification, regression, pipelines
     - (included)
   * - sktime
     - ``ForecastingOptCV``
     - Time series forecasting
     - ``[sktime-integration]``
   * - sktime
     - ``TSCOptCV``
     - Time series classification
     - ``[sktime-integration]``
   * - skpro
     - ``SkproProbaRegExperiment``
     - Probabilistic regression
     - ``[all_extras]``
   * - PyTorch
     - ``TorchExperiment``
     - Deep learning models
     - ``[all_extras]``

----

Scikit-Learn Integration
------------------------

The ``OptCV`` class provides a scikit-learn compatible interface for hyperparameter
tuning. It works like ``GridSearchCV`` but supports any Hyperactive optimizer.

.. grid:: 1
   :gutter: 0

   .. grid-item::
      :class: sd-bg-light sd-pt-3 sd-pb-1 sd-ps-3 sd-pe-3 sd-rounded-3

      **Key Features**

      - Drop-in replacement for ``GridSearchCV``
      - Works with any sklearn estimator or pipeline
      - Use any Hyperactive optimizer
      - Full cross-validation support

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

----

Time Series with Sktime
-----------------------

Hyperactive integrates with ``sktime`` for time series forecasting and classification.

.. note::

   Requires ``pip install hyperactive[sktime-integration]``

.. grid:: 1
   :gutter: 0

   .. grid-item::
      :class: sd-bg-light sd-pt-3 sd-pb-1 sd-ps-3 sd-pe-3 sd-rounded-3

      **Key Features**

      - Optimize forecasters with temporal cross-validation
      - Time series classification support
      - Compatible with sktime's model interface


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

----

Probabilistic Prediction with Skpro
-----------------------------------

For probabilistic regression with ``skpro``:

.. grid:: 1
   :gutter: 0

   .. grid-item::
      :class: sd-bg-light sd-pt-3 sd-pb-1 sd-ps-3 sd-pe-3 sd-rounded-3

      **Key Features**

      - Optimize probabilistic regressors
      - Proper scoring rules for uncertainty quantification
      - Full skpro compatibility

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:skpro_experiment]
   :end-before: # [end:skpro_experiment]

----

PyTorch Lightning Integration
-----------------------------

For deep learning hyperparameter optimization with PyTorch Lightning:

.. note::

   Requires ``pip install hyperactive[all_extras]`` or ``pip install lightning``

.. grid:: 1
   :gutter: 0

   .. grid-item::
      :class: sd-bg-light sd-pt-3 sd-pb-1 sd-ps-3 sd-pe-3 sd-rounded-3

      **Key Features**

      - Optimize neural network architectures
      - Tune training hyperparameters (learning rate, batch size, etc.)
      - Early stopping and pruning support

.. literalinclude:: ../_snippets/user_guide/integrations.py
   :language: python
   :start-after: # [start:pytorch_lightning]
   :end-before: # [end:pytorch_lightning]

----

Tips
----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Match the interface

      Use ``OptCV`` when you want sklearn-compatible behavior (fit/predict).
      Use experiment classes when you want more control over the optimization loop.

   .. grid-item-card:: Consider evaluation cost

      Deep learning experiments are expensive. Use efficient optimizers like
      ``BayesianOptimizer`` with fewer iterations (10-50 instead of 100+).

   .. grid-item-card:: Use appropriate CV strategies

      Match your cross-validation to your problem: ``TimeSeriesSplit`` for
      time series, stratified splits for imbalanced data.

   .. grid-item-card:: Start simple

      Begin with ``RandomSearch`` to establish baselines before using
      more sophisticated optimizers.
