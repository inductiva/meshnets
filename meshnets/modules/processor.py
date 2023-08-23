""""Define the Processor and ProcesserLayer classes."""

from typing import Tuple

import torch
from torch_geometric.nn.conv import MessagePassing
import torch_scatter

from meshnets.modules.mlp import MLP


class MGNProcessorLayer(MessagePassing):
    """Single MGN Message Passing layer for graphs."""

    def __init__(self, latent_size, num_mlp_layers):
        """Build the MLP encoders for message passing.

        Note that the latent node features and latent edge features
        have the same size. Thus, the input of the edge processor is
        three times the hidden dimension (one self embedding and two
        adjacent node embeddings). The input of the node processor is
        two times the hidden dimension (one self embeding and one 
        aggregation of connected edges embedings)
        """
        super().__init__()

        edge_mlp_widths = [3 * latent_size
                          ] + (num_mlp_layers + 1) * [latent_size]
        self.edge_mlp = MLP(edge_mlp_widths, layer_norm=True)

        node_mlp_widths = [2 * latent_size
                          ] + (num_mlp_layers + 1) * [latent_size]
        self.node_mlp = MLP(node_mlp_widths, layer_norm=True)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate information on edges, then compute updated node features.
        
        Returns the updated edge and node features."""

        aggregated_edges, updated_edges = self.propagate(edge_index=edge_index,
                                                         x=x,
                                                         edge_attr=edge_attr)

        # Update the nodes with a residual connection
        updated_nodes = torch.cat([x, aggregated_edges], dim=1)
        updated_nodes = self.node_mlp(updated_nodes) + x

        # update the graph features
        x = updated_nodes
        edge_attr = updated_edges

        return x, edge_attr

    def message(self, x_i, x_j, edge_attr):

        # Update the edges with a residual connection
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        # Sum the updated edges features for each node
        _, target = edge_index
        aggregated_edges = torch_scatter.scatter_sum(updated_edges,
                                                     target,
                                                     dim=0)

        return aggregated_edges, updated_edges

    def __repr__(self) -> str:
        """Use torch.nn.Module representation."""
        return torch.nn.Module.__repr__(self)


class MGNProcessor(torch.nn.Module):
    """MGN Processor for graphs.
    
    It is a sequence of message-passing MGNProcessorLayers applied to the
    the latent node and mesh features."""

    def __init__(self, latent_size, num_mlp_layers, message_passing_steps):
        """Initialize the processor."""
        super().__init__()

        self._latent_size = latent_size
        self._num_mlp_layers = num_mlp_layers
        self._message_passing_steps = message_passing_steps

        self.processor_layers = self._build_processor()

    def _build_processor(self):
        """Build the GraphProcessor as a list of ProcessorLayer
        with the given latent size and number of MLP layers."""

        processor_layers = torch.nn.ModuleList([
            MGNProcessorLayer(latent_size=self._latent_size,
                              num_mlp_layers=self._num_mlp_layers)
            for _ in range(self._message_passing_steps)
        ])

        return processor_layers

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the GraphProcessor to graph data.
        
        Return the processed features."""

        for layer in self.processor_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr
