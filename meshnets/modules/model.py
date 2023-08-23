"""Define the main MeshGraphNet model"""

from typing import Tuple

import torch
from torch_geometric.data import Batch

from meshnets.modules.encoder import GraphEncoder
from meshnets.modules.decoder import GraphDecoder
from meshnets.modules.processor import MGNProcessor


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNet model.
    
    The model architecture is composed of a GraphEncoder, a GraphProcessor,
    a GraphDecoder, and takes as input graph objects. Each graph features
    are encoded to form a latent graph representation. Message passing steps
    are then applied by the Processor on the latent graphs and the Decoder
    outputs features at each node from the processed latent node features."""

    def __init__(self, node_features_size, edge_features_size, output_size,
                 latent_size, num_mlp_layers, message_passing_steps, x_mean,
                 x_std, edge_attr_mean, edge_attr_std):
        """Initialize the input statistics, GraphEncoder, GraphProcessor
        and GraphDecoder of the MeshGraphNet."""
        super().__init__()

        self.x_mean = x_mean
        self.x_std = x_std
        self.edge_attr_mean = edge_attr_mean
        self.edge_attr_std = edge_attr_std

        self.encoder = GraphEncoder(node_features_size, edge_features_size,
                                    latent_size, num_mlp_layers)

        self.processor = MGNProcessor(latent_size, num_mlp_layers,
                                      message_passing_steps)

        self.decoder = GraphDecoder(output_size, latent_size, num_mlp_layers)

    def normalize_input(
            self, x: torch.Tensor,
            edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize the graph features.
        
        The node and edge features are normalized according to the mean and std
        given to the model."""

        # Send the stats to the same device as the data and normalize
        x = (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

        edge_attr = (edge_attr - self.edge_attr_mean.to(
            edge_attr.device)) / self.edge_attr_std.to(edge_attr.device)

        return x, edge_attr

    def forward(self, data: Batch) -> torch.Tensor:
        """Apply the MeshGraphNet to a batch of graphs
        and return the prediction tensor."""

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x, edge_attr = self.normalize_input(x, edge_attr)

        x, edge_attr = self.encoder(x, edge_attr)
        x, edge_attr = self.processor(x, edge_index, edge_attr)
        prediction = self.decoder(x)

        return prediction
