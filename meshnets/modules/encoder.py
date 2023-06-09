""""Define the GraphEncoder class."""

import torch
from torch_geometric.data import Batch

from meshnets.modules.mlp import MLP


class GraphEncoder(torch.nn.Module):
    """Encoder for graphs.
    
    Encode graphs to latent graphs by applying an MLP encoder to each
    node features and a second MLP encoder to each mesh features.
    """

    def __init__(self, node_feats_size, mesh_feats_size, latent_size,
                 num_mlp_layers):
        """Initialize the node features encoder and the mesh features encoder
        given the features sizes, latent size and number of MLP layers."""
        super().__init__()

        node_encoder_widths = [node_feats_size
                              ] + (num_mlp_layers + 1) * [latent_size]
        self.node_encoder = MLP(node_encoder_widths, layer_norm=True)

        edge_encoder_widths = [mesh_feats_size
                              ] + (num_mlp_layers + 1) * [latent_size]
        self.edge_encoder = MLP(edge_encoder_widths, layer_norm=True)

    def forward(self, graph: Batch) -> Batch:
        """Encode the node and mesh features of a batch of graphs
        by applying the corresponding MLP encoder to each feature.
        
        Return the enconded batch."""

        graph.x = self.node_encoder(graph.x)
        graph.edge_attr = self.edge_encoder(graph.edge_attr)

        return graph
