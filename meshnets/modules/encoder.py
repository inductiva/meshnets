""""Define the GraphEncoder class."""

import torch

class GraphEncoder(torch.nn.Module):

    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, graph):
        return graph
