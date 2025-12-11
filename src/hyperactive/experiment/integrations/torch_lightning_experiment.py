"""Experiment adapter for PyTorch Lightning experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

__author__ = ["amitsubhashchejara"]

import numpy as np

from hyperactive.base import BaseExperiment


class TorchExperiment(BaseExperiment):
    """Experiment adapter for PyTorch Lightning experiments.

    This class is used to perform experiments using PyTorch Lightning modules.
    It allows for hyperparameter tuning and evaluation of the model's performance
    using specified metrics.

    The experiment trains a Lightning module with given hyperparameters and returns
    the validation metric value for optimization.

    Parameters
    ----------
    data_module : type
        A PyTorch Lightning DataModule class (not an instance) that
        handles data loading and preparation. It will be instantiated
        with hyperparameters during optimization.
    lightning_module : type
        A PyTorch Lightning Module class (not an instance) that will be instantiated
        with hyperparameters during optimization.
    trainer_kwargs : dict, optional (default=None)
        A dictionary of keyword arguments to pass to the PyTorch Lightning Trainer.
    objective_metric : str, optional (default='val_loss')
        The metric used to evaluate the model's performance. This should correspond
        to a metric logged in the LightningModule during validation.

    Examples
    --------
    >>> from hyperactive.experiment.integrations import TorchExperiment
    >>> import torch
    >>> import lightning as L
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Define a simple Lightning Module
    >>> class SimpleLightningModule(L.LightningModule):
    ...     def __init__(self, input_dim=10, hidden_dim=16, lr=1e-3):
    ...         super().__init__()
    ...         self.save_hyperparameters()
    ...         self.model = nn.Sequential(
    ...             nn.Linear(input_dim, hidden_dim),
    ...             nn.ReLU(),
    ...             nn.Linear(hidden_dim, 2)
    ...         )
    ...         self.lr = lr
    ...
    ...     def forward(self, x):
    ...         return self.model(x)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         loss = nn.functional.cross_entropy(y_hat, y)
    ...         self.log("train_loss", loss)
    ...         return loss
    ...
    ...     def validation_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         val_loss = nn.functional.cross_entropy(y_hat, y)
    ...         self.log("val_loss", val_loss, on_epoch=True)
    ...         return val_loss
    ...
    ...     def configure_optimizers(self):
    ...         return torch.optim.Adam(self.parameters(), lr=self.lr)
    >>>
    >>> # Create DataModule
    >>> class RandomDataModule(L.LightningDataModule):
    ...     def __init__(self, batch_size=32):
    ...         super().__init__()
    ...         self.batch_size = batch_size
    ...
    ...     def setup(self, stage=None):
    ...         dataset = torch.utils.data.TensorDataset(
    ...             torch.randn(100, 10),
    ...             torch.randint(0, 2, (100,))
    ...         )
    ...         self.train, self.val = torch.utils.data.random_split(
    ...             dataset, [80, 20]
    ...         )
    ...
    ...     def train_dataloader(self):
    ...         return DataLoader(self.train, batch_size=self.batch_size)
    ...
    ...     def val_dataloader(self):
    ...         return DataLoader(self.val, batch_size=self.batch_size)
    >>>
    >>> # Create Experiment
    >>> experiment = TorchExperiment(
    ...     data_module=RandomDataModule,
    ...     lightning_module=SimpleLightningModule,
    ...     trainer_kwargs={'max_epochs': 3},
    ...     objective_metric="val_loss"
    ... )
    >>>
    >>> lm_params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}
    >>> dm_params = {"batch_size": 32}
    >>>
    >>> val_result, metadata = experiment._evaluate(lm_params, dm_params)
    """

    _tags = {
        "property:randomness": "random",
        "property:higher_or_lower_is_better": "lower",
        "authors": ["amitsubhashchejara"],
        "python_dependencies": ["torch", "lightning"],
    }

    def __init__(
        self,
        data_module,
        lightning_module,
        trainer_kwargs=None,
        objective_metric: str = "val_loss",
    ):
        self.data_module = data_module
        self.lightning_module = lightning_module
        self.trainer_kwargs = trainer_kwargs or {}
        self.objective_metric = objective_metric

        super().__init__()

        self._trainer_kwargs = {
            "max_epochs": 10,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        if trainer_kwargs is not None:
            self._trainer_kwargs.update(trainer_kwargs)

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str, or None
            The parameter names of the search parameters.
            If not known or arbitrary, return None.
        """
        import inspect

        sig = inspect.signature(self.lightning_module.__init__)
        return [p for p in sig.parameters.keys() if p != "self"]

    def _evaluate(self, lm_params, dm_params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        import lightning as L

        try:
            model = self.lightning_module(**lm_params)
            trainer = L.Trainer(**self._trainer_kwargs)
            data = self.data_module(**dm_params)
            trainer.fit(model, data)

            val_result = trainer.callback_metrics.get(self.objective_metric)
            metadata = {}

            if val_result is None:
                available_metrics = list(trainer.callback_metrics.keys())
                raise ValueError(
                    f"Metric '{self.objective_metric}' not found. "
                    f"Available: {available_metrics}"
                )
            if hasattr(val_result, "item"):
                val_result = np.float64(val_result.detach().cpu().item())
            elif isinstance(val_result, int | float):
                val_result = np.float64(val_result)
            else:
                val_result = np.float64(float(val_result))

            return val_result, metadata

        except Exception as e:
            print(f"Training failed with params {lm_params}: {e}")
            return np.float64(float("inf")), {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        import lightning as L
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        class SimpleLightningModule(L.LightningModule):
            def __init__(self, input_dim=10, hidden_dim=16, lr=1e-3):
                super().__init__()
                self.save_hyperparameters()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2),
                )
                self.lr = lr

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = nn.functional.cross_entropy(y_hat, y)
                self.log("train_loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                val_loss = nn.functional.cross_entropy(y_hat, y)
                self.log("val_loss", val_loss, on_epoch=True)
                return val_loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr)

        class RandomDataModule(L.LightningDataModule):
            def __init__(self, batch_size=32):
                super().__init__()
                self.batch_size = batch_size

            def setup(self, stage=None):
                dataset = torch.utils.data.TensorDataset(
                    torch.randn(100, 10), torch.randint(0, 2, (100,))
                )
                self.train, self.val = torch.utils.data.random_split(dataset, [80, 20])

            def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.batch_size)

            def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.batch_size)

        params = {
            "data_module": RandomDataModule,
            "lightning_module": SimpleLightningModule,
            "trainer_kwargs": {
                "max_epochs": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
            "objective_metric": "val_loss",
        }

        class RegressionModule(L.LightningModule):
            def __init__(self, num_layers=2, hidden_size=32, dropout=0.1):
                super().__init__()
                self.save_hyperparameters()
                layers = []
                input_size = 20
                for _ in range(num_layers):
                    layers.extend(
                        [
                            nn.Linear(input_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ]
                    )
                    input_size = hidden_size
                layers.append(nn.Linear(hidden_size, 1))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x).squeeze()
                loss = nn.functional.mse_loss(y_hat, y)
                self.log("train_loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x).squeeze()
                val_loss = nn.functional.mse_loss(y_hat, y)
                self.log("val_loss", val_loss, on_epoch=True)
                return val_loss

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        class RegressionDataModule(L.LightningDataModule):
            def __init__(self, batch_size=16, num_samples=150):
                super().__init__()
                self.batch_size = batch_size
                self.num_samples = num_samples

            def setup(self, stage=None):
                X = torch.randn(self.num_samples, 20)
                y = torch.randn(self.num_samples)
                dataset = torch.utils.data.TensorDataset(X, y)
                train_size = int(0.8 * self.num_samples)
                val_size = self.num_samples - train_size
                self.train, self.val = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

            def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.batch_size)

            def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.batch_size)

        params2 = {
            "data_module": RegressionDataModule,
            "lightning_module": RegressionModule,
            "trainer_kwargs": {
                "max_epochs": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
            "objective_metric": "val_loss",
        }

        return [params, params2]

    @classmethod
    def _get_score_params(cls):
        """Return settings for testing score/evaluate functions.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        score_lm_params1 = {"input_dim": 10, "hidden_dim": 20, "lr": 0.001}
        score_lm_params2 = {"num_layers": 3, "hidden_size": 64, "dropout": 0.2}
        score_dm_params1 = {"batch_size": 16}
        score_dm_params2 = {"batch_size": 8, "num_samples": 200}

        return [
            {"lm_params": score_lm_params1, "dm_params": score_dm_params1},
            {"lm_params": score_lm_params2, "dm_params": score_dm_params2},
        ]
