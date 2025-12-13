.. _troubleshooting_experiments:

===========================
Experiment-Specific Issues
===========================

SklearnCvExperiment Not Finding Best Parameters
------------------------------------------------

**Cause**: Search space doesn't include good values or not enough iterations.

**Solutions**:

1. Verify search space includes reasonable values:

   .. code-block:: python

       # Make sure these are sensible for your model
       search_space = {
           "n_estimators": [10, 50, 100, 200, 500],
           "max_depth": [None, 3, 5, 10, 20],
       }

2. Increase iterations or use smarter optimizer:

   .. code-block:: python

       optimizer = BayesianOptimizer(
           search_space=space,
           n_iter=200,  # More iterations
           experiment=experiment,
       )


PyTorch Lightning Metric Not Found
----------------------------------

**Cause**: The metric name doesn't match what's logged during training.

**Solution**: Check your Lightning module logs the correct metric:

.. code-block:: python

    class MyModel(L.LightningModule):
        def validation_step(self, batch, batch_idx):
            loss = self.compute_loss(batch)
            self.log("val_loss", loss)  # Must match objective_metric

    experiment = TorchLightningExperiment(
        lightning_module=MyModel,
        objective_metric="val_loss",  # Must match self.log name
        ...
    )
