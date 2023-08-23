""""Define the GraphDecoder class."""

import torch

from meshnets.modules.mlp import MLP


class GraphDecoder(torch.nn.Module):
    """Decoder for graphs.
    
    Decode latent graphs to an output Tensor corresponding to predicted
    node features by applying an MLP decoder to each node features.
    """

    def __init__(self, output_size, latent_size, num_mlp_layers):
        """Initialize the node features decoder given the output size,
        latent size and number of MLP layers."""
        super().__init__()

        decoder_widths = (num_mlp_layers + 1) * [latent_size] + [output_size]
        self.node_decoder = MLP(decoder_widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the latent node features of a batch of graphs
        to output node features.
        
        Return the output Tensor."""

        return self.node_decoder(x)
