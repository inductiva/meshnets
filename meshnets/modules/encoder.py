""""Define the GraphEncoder class."""

import torch
from torch.nn import Identity

class GraphEncoder(torch.nn.Module):
    """"GraphEncoder class."""

    def __init__(self, node_features_size, mesh_features_size,
                 latent_size, num_mlp_layers):
        super().__init__()

        self.module = Identity(node_features_size,
                               mesh_features_size,
                               latent_size,
                               num_mlp_layers)

    def forward(self, graph):
        return self.module(graph)
