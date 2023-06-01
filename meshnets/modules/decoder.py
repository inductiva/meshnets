""""Define the GraphDecoder class."""

import torch
from torch.nn import Linear
from torch_geometric.data import Batch


class GraphDecoder(torch.nn.Module):
    """Decoder for graphs.
    
    Decode latent graphs to an output Tensor corresponding to predicted
    node features by applying an MLP decoder to each node features.
    """

    # pylint: disable=unused-argument
    def __init__(self, output_size, latent_size, num_mlp_layers):
        """Initialize the node features decoder given the output size,
        latent size and number of MLP layers."""
        super().__init__()

        # TODO(victor) : implement decoder
        self.node_decoder = Linear(latent_size, output_size)

    def forward(self, graph: Batch) -> torch.Tensor:
        """Decode the latent node features of a batch of graphs
        to output node features.
        
        Return the output Tensor."""

        return self.node_decoder(graph.x)
