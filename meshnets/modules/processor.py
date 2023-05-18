""""Define the GraphProcessor class."""

import torch
from torch.nn import Identity

class GraphProcessor(torch.nn.Module):

    def __init__(self, latent_size, message_passing_steps):
        super().__init__()

        self.module = Identity(latent_size,
                               message_passing_steps)

    def forward(self, graph):
        return self.module(graph)
