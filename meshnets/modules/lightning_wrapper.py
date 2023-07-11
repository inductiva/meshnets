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
        super().__init__()
        self.model = model
        self.data_stats = data_stats
        self.learning_rate = learning_rate

    def normalize(self, batch: Batch) -> Batch:

        # Normalize the input features
        batch.x = (batch.x -
                   self.data_stats['x_mean']) / self.data_stats['x_std']

        batch.edge_attr = (batch.edge_attr - self.data_stats['edge_attr_mean']
                          ) / self.data_stats['edge_attr_std']

        # Normalize the labels
        if batch.y is not None:
            batch.y = (batch.y -
                       self.data_stats['y_mean']) / self.data_stats['y_std']

        return batch

    def forward(self, batch: Batch) -> torch.Tensor:
        batch = self.normalize(batch)
        return self.model(batch)

    def training_step(self, batch: Batch, _) -> dict:

        batch = self.normalize(batch)

        loss = self.compute_loss(batch)
        # batch_size can vary depending on the number of nodes in the graphs
        # so we pass it explicitly to the logger
        # TODO(victor): check that batch_size is set correctly or if it
        # should be the number of graphs in the batch
        self.log('loss', loss, on_epoch=True, batch_size=batch.num_nodes)
        return {'loss': loss}

    def validation_step(self, batch: Batch, _) -> dict:

        batch = self.normalize(batch)

        val_loss = self.compute_loss(batch)
        # batch_size can vary depending on the number of nodes in the graphs
        # so we pass it explicitly to the logger
        # TODO(victor): check that batch_size is set correctly or if it
        # should be the number of graphs in the batch
        self.log('val_loss',
                 val_loss,
                 on_epoch=True,
                 batch_size=batch.num_nodes,
                 prog_bar=True)
        return {'val_loss': val_loss}

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        predictions = self.model(batch)
        loss = torch.nn.functional.mse_loss(predictions, batch.y)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
