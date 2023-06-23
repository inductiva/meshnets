""""Define the Processor and ProcesserLayer classes."""

import torch
from torch.nn import Sequential
from torch_geometric.data import Batch
from torch_geometric.nn.conv import MessagePassing
import torch_scatter

from meshnets.modules.mlp import MLP


class MGNProcessorLayer(MessagePassing):
    """Single MGN Message Passing layer for graphs."""

    def __init__(self, latent_size, num_mlp_layers):
        """Initialize a processor layer given the latent size
        and number of layers in the MLPs.

        Note that the latent node features and latent edge features
        have the same size. Thus, the input of the edge processor is
        three times the hidden dimension (one self embedding and two
        adjacent node embeddings). The input of the node processor is
        two times the hidden dimension (one self embedding and one sum
        of connected edges embedings)
        """
        super().__init__()

        edge_mlp_widths = [3 * latent_size
                          ] + (num_mlp_layers + 1) * [latent_size]
        self.edge_mlp = MLP(edge_mlp_widths, layer_norm=True)

        node_mlp_widths = [2 * latent_size
                          ] + (num_mlp_layers + 1) * [latent_size]
        self.node_mlp = MLP(node_mlp_widths, layer_norm=True)

    def forward(self, graph: Batch) -> Batch:
        """
        Call the propagate method to initiate message passing and aggregation
        on edges, and then compute the updated node features.
        
        It returns the graph with updated edge and node features."""

        aggregated_edges, updated_edges = self.propagate(
            edge_index=graph.edge_index, x=graph.x, edge_attr=graph.edge_attr)

        # Update the nodes with a residual connection
        updated_nodes = torch.cat([graph.x, aggregated_edges], dim=1)
        updated_nodes = self.node_mlp(updated_nodes) + graph.x

        # update the graph features
        graph.x = updated_nodes
        graph.edge_attr = updated_edges

        return graph

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


class MGNProcessor(torch.nn.Module):
    """MGN Processor for graphs.
    
    It is a sequence of message-passing MGNProcessorLayers applied to the
    the latent node and mesh features."""

    def __init__(self, latent_size, num_mlp_layers, message_passing_steps):
        """Initialize the processor given the latent size, number of layers
        in the MLPs and number of message passing steps."""
        super().__init__()

        self._latent_size = latent_size
        self._num_mlp_layers = num_mlp_layers
        self._message_passing_steps = message_passing_steps

        self.processor = self._build_processor()

    def _build_processor(self):
        """Build the GraphProcessor as a sequence of ProcessorLayer
        with the given latent size and number of layers."""

        layers = []
        for _ in range(self._message_passing_steps):
            layers.append(
                MGNProcessorLayer(self._latent_size, self._num_mlp_layers))

        return Sequential(*layers)

    def forward(self, graph: Batch) -> Batch:
        """Apply the GraphProcessor to a batch of graphs.
        
        Return the batch with processed features."""
        return self.processor(graph)
