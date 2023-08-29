"""Define lightning wrappers."""

from typing import Type
from typing import Sequence

import torch
from torch_geometric.data import Batch
import pytorch_lightning as pl


class MGNLightningWrapper(pl.LightningModule):
    """Ppytorch_lightning wrapper for the MeshGraphNet training logic."""

    def __init__(self,
                 model: Type[torch.nn.Module],
                 y_mean: torch.Tensor,
                 y_std: torch.Tensor,
                 validation_datasets_names: Sequence[str],
                 learning_rate: float = 1e-3,
                 **model_args):
        """Initialize for a given model, its arguments, label statistics,
        and a learning rate.
        """
        super().__init__()

        self.save_hyperparameters()

        self.y_mean = y_mean
        self.y_std = y_std

        self.model = model(**model_args)
        self.learning_rate = learning_rate

        self.validation_datasets_names = validation_datasets_names

    @torch.no_grad()
    def normalize_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize the labels in a batch.
        
        The labels are normalized according to the mean and std given
        to the wrapper."""

        # Send the stats to the same device as the data and normalize
        y = (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

        return y

    @torch.no_grad()
    def unnormalize_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Unnormalize the labels in a batch.
        
        The labels are unnormalized according to the mean and std given
        to the wrapper."""

        # Send the stats to the same device as the data and normalize
        y = (y * self.y_std.to(y.device)) + self.y_mean.to(y.device)

        return y

    def forward(self, batch: Batch) -> torch.Tensor:
        """Make a forward pass in the model and unnormalize prediction."""
        prediction = self.model(batch)
        return self.unnormalize_labels(prediction)

    def training_step(self, batch: Batch, _) -> dict:
        loss = self.compute_loss(batch)
        # batch_size is set to the number of nodes in the batch in order
        # to weight the batch losses correctly
        self.log('loss', loss, on_epoch=True, batch_size=batch.num_nodes)
        return {'loss': loss}

    def validation_step(self, batch: Batch, _, dataloader_idx: int = 0) -> dict:
        val_loss = self.compute_loss(batch)
        # batch_size is set to the number of nodes in the batch in order
        # to weight the batch losses correctly
        val_dataset_name = self.validation_datasets_names[dataloader_idx]
        self.log(f'val_loss_{val_dataset_name}',
                 val_loss,
                 on_epoch=True,
                 batch_size=batch.num_nodes,
                 prog_bar=True)
        return {'val_loss': val_loss}

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        # Make a forward pass in the model
        predictions = self.model(batch)
        # Normalize the labels before computing the loss
        y_norm = self.normalize_labels(batch.y)
        loss = torch.nn.functional.mse_loss(predictions, y_norm)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
