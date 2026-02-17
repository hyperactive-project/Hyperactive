.. _examples_skforecast_integration:

=====================
Skforecast Integration
=====================

Hyperactive integrates with ``skforecast`` to tune forecasting models with any
Hyperactive optimizer.


Example File
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Forecasting with ``SkforecastOptCV``
     - `skforecast_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/skforecast/skforecast_example.py>`_


Installation
------------

Install the optional integration dependency:

.. code-block:: bash

    pip install hyperactive[skforecast-integration]

.. note::

   ``skforecast`` currently requires Python < 3.14.


Usage Overview
--------------

The example below shows the sklearn-like ``fit``/``predict`` workflow with
``SkforecastOptCV``:

.. literalinclude:: ../../../examples/skforecast/skforecast_example.py
   :language: python


See :ref:`user_guide_integrations` for the full integration overview.
