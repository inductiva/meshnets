""""Define the Processor and ProcesserLayer classes."""

import torch
from torch.nn import Identity
from torch_geometric.data import Batch
from torch_geometric.nn.conv import MessagePassing


class MGNProcessorLayer(MessagePassing):
    """Single Message Passing layer for a Graph object."""

    def __init__(self, latent_size):
        super().__init__()

        # TODO(victor) : implement layer
        self.layer = Identity(latent_size)

    def forward(self, graph: Batch) -> Batch:
        return self.layer(graph)


class MGNProcessor(torch.nn.Module):
    """Processor for latent Graph object.
    
    It is a sequence of message-passing ProcessorLayers applied to the
    the latent node and mesh features."""

    def __init__(self, latent_size, message_passing_steps):
        """Initialize the processor given the latent size
        and number of message passing steps"""
        super().__init__()

        # TODO(victor) : implement processor
        self.processor = Identity(latent_size, message_passing_steps)

    def forward(self, graph: Batch) -> Batch:
        """Apply the GraphProcessor to a Graph object.
        
        Return a Graph object with processed features."""
        return self.processor(graph)
