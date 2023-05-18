""""Define the GraphDecoder class."""

import torch

class GraphDecoder(torch.nn.Module):

    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, graph):
        return graph
