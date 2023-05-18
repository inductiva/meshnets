""""Define the GraphDecoder class."""

import torch
from torch.nn import Identity

class GraphDecoder(torch.nn.Module):
    """"GraphDecoder class."""

    def __init__(self, output_size, latent_size, num_mlp_layers):
        super().__init__()

        self.module = Identity(output_size,
                               latent_size,
                               num_mlp_layers)

    def forward(self, graph):
        return self.module(graph)
