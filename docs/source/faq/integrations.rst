.. _faq_integrations:

============
Integrations
============

Can I use Hyperactive with PyTorch (not Lightning)?
---------------------------------------------------

Yes, create a custom objective function:

.. code-block:: python

    import torch

    def objective(params):
        model = MyPyTorchModel(
            hidden_size=params["hidden_size"],
            dropout=params["dropout"],
        )
        # Train and evaluate your model
        train_model(model, train_loader)
        accuracy = evaluate_model(model, val_loader)
        return accuracy


How does Hyperactive compare to Optuna?
---------------------------------------

**Hyperactive with native GFO backend**:

- Simple, unified API
- Wide variety of optimization algorithms
- Great for hyperparameter tuning

**Hyperactive with Optuna backend**:

- Access Optuna's algorithms through Hyperactive's interface
- Combine the strengths of both libraries

**Pure Optuna**:

- More features (pruning, distributed, database storage)
- Larger community and ecosystem
- More configuration options

Choose based on your needs: Hyperactive for simplicity, Optuna for
advanced features.


Can I use Hyperactive with other ML frameworks?
-----------------------------------------------

Yes, any framework works with custom objective functions:

.. code-block:: python

    # XGBoost example
    import xgboost as xgb

    def objective(params):
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3)
        return scores.mean()
