"""Example: Hyperparameter tuning with TorchExperiment and PyTorch Lightning.

This example demonstrates how to use the TorchExperiment class to
optimize hyperparameters of a PyTorch Lightning model using Hyperactive.
"""

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hyperactive.experiment.integrations import TorchExperiment
from hyperactive.opt.gfo import HillClimbing


# 1. Define a Lightning Module
class SimpleLightningModule(L.LightningModule):
    """Simple classification model for demonstration."""

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
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate a single batch."""
        x, y = batch
        y_hat = self(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# 2. Define a DataModule
class RandomDataModule(L.LightningDataModule):
    """Random data module for demonstration."""

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        """Set up train and validation datasets."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(200, 10),
            torch.randint(0, 2, (200,)),
        )
        self.train, self.val = torch.utils.data.random_split(dataset, [160, 40])

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size)


# 3. Create the TorchExperiment
datamodule = RandomDataModule(batch_size=16)
datamodule.setup()

experiment = TorchExperiment(
    datamodule=datamodule,
    lightning_module=SimpleLightningModule,
    trainer_kwargs={
        "max_epochs": 3,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    },
    objective_metric="val_loss",
)

# 4. Define search space and optimizer
search_space = {
    "hidden_dim": [16, 32, 64, 128],
    "lr": np.logspace(-4, -1, 10).tolist(),
}

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=5,
    experiment=experiment,
)

# 5. Run optimization
best_params = optimizer.solve()
print(f"Best params: {best_params}")
