""""Define the GraphEncoder class."""

import torch
from torch.nn import Identity
from torch_geometric.data import Batch


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

        # TODO(victor) : implement encoders
        self.node_encoder = Identity(node_feats_size, latent_size,
                                     num_mlp_layers)
        self.edge_encoder = Identity(mesh_feats_size, latent_size,
                                     num_mlp_layers)

    def forward(self, graph: Batch) -> Batch:
        """Encode the node and mesh features of a batch of graphs
        by applying the corresponding MLP encoder to each feature.
        
        Return the enconded batch."""

        graph.node_features = self.node_encoder(graph.node_features)
        graph.edge_set.features = self.edge_encoder(graph.edge_set.features)

        return graph
