.. _examples_integrations:

============
Integrations
============

Hyperactive integrates with popular machine learning frameworks beyond
scikit-learn, including time series libraries.


Sktime Integration
------------------

For time series forecasting and classification with sktime:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Time Series Forecasting
     - `sktime_forecasting_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sktime_forecasting_example.py>`_
   * - Time Series Classification
     - `sktime_tsc_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sktime_tsc_example.py>`_


Skforecast Integration
----------------------

For direct ``skforecast`` forecasting optimization:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Recursive forecasting with ``SkforecastOptCV``
     - `skforecast_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/skforecast/skforecast_example.py>`_

.. note::

   Sktime integration requires additional dependencies:

   .. code-block:: bash

       pip install hyperactive[sktime-integration]


Installing Extras
-----------------

Install integration extras as needed:

.. code-block:: bash

    # Sktime/skpro for time series
    pip install hyperactive[sktime-integration]

    # Skforecast integration
    pip install hyperactive[skforecast-integration]

    # All extras including PyTorch Lightning
    pip install hyperactive[all_extras]

See :ref:`user_guide_integrations` for complete integration documentation.
