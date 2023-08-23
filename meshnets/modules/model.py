"""Define the main MeshGraphNet model"""

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
                 latent_size, num_mlp_layers, message_passing_steps):
        """Initialize the GraphEncoder, GraphProcessor and GraphDecoder
        of the MeshGraphNet."""
        super().__init__()

        self.encoder = GraphEncoder(node_features_size, edge_features_size,
                                    latent_size, num_mlp_layers)

        self.processor = MGNProcessor(latent_size, num_mlp_layers,
                                      message_passing_steps)

        self.decoder = GraphDecoder(output_size, latent_size, num_mlp_layers)

    def forward(self, data: Batch) -> torch.Tensor:
        """Apply the MeshGraphNet to a batch of graphs
        and return the prediction tensor."""

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x, edge_attr = self.encoder(x, edge_attr)
        x, edge_attr = self.processor(x, edge_index, edge_attr)
        prediction = self.decoder(x)

        return prediction
