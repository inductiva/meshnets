""""Define the GraphProcessor class."""

import torch

class GraphProcessor(torch.nn.Module):

    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, graph):
        return graph
