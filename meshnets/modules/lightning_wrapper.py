"""Define lightning wrappers."""

import torch
from torch_geometric.data import Batch
import pytorch_lightning as pl


class MGNLightningWrapper(pl.LightningModule):
    """Simple pytorch_lightning wrapper for the MeshGraphNet training logic."""

    def __init__(self,
                 model: torch.nn.Module,
                 data_stats: dict,
                 learning_rate: float = 1e-3):
        """Initialize for a given model, a dictionnary of
        data statistics for normalization, and a learning rate.
        
        The statistics dictionnary is expected to have
        the following fields:
            'x_mean'
            'x_std'
            'edge_attr_mean'
            'eadge_attr_std'
            'y_mean'
            'y_std'
        """
        super().__init__()

        self.model = model
        self.data_stats = data_stats
        self.learning_rate = learning_rate

    def normalize_input(self, batch: Batch) -> Batch:
        """Normalize the input features in a batch.
        
        The node and edge features are normalized using the mean and std
        statistics found in the data_stats dictionnary."""

        # Send the stats to the same device as the data and normalize
        batch.x = (batch.x - self.data_stats['x_mean'].to(
            batch.x.device)) / self.data_stats['x_std'].to(batch.x.device)

        batch.edge_attr = (
            batch.edge_attr - self.data_stats['edge_attr_mean'].to(
                batch.edge_attr.device)) / self.data_stats['edge_attr_std'].to(
                    batch.edge_attr.device)

        return batch

    def normalize_labels(self, batch: Batch) -> Batch:
        """Normalize the labels in a batch.
        
        The labels are normalized using the mean and std statistics
        found in the data_stats dictionnary."""

        # Send the stats to the same device as the data and normalize
        batch.y = (batch.y - self.data_stats['y_mean'].to(
            batch.y.device)) / self.data_stats['y_std'].to(batch.y.device)

        return batch

    def forward(self, batch: Batch) -> torch.Tensor:
        """Normalize the batch before making a forward pass in the model."""
        batch = self.normalize_input(batch)
        return self.model(batch)

    def training_step(self, batch: Batch, _) -> dict:
        loss = self.compute_loss(batch)
        # batch_size can vary depending on the number of nodes in the graphs
        # so we pass it explicitly to the logger
        self.log('loss', loss, on_epoch=True, batch_size=batch.num_nodes)
        return {'loss': loss}

    def validation_step(self, batch: Batch, _) -> dict:
        val_loss = self.compute_loss(batch)
        # batch_size can vary depending on the number of nodes in the graphs
        # so we pass it explicitly to the logger
        self.log('val_loss',
                 val_loss,
                 on_epoch=True,
                 batch_size=batch.num_nodes,
                 prog_bar=True)
        return {'val_loss': val_loss}

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        predictions = self.forward(batch)
        batch = self.normalize_labels(batch)
        loss = torch.nn.functional.mse_loss(predictions, batch.y)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
