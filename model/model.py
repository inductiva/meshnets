"""Define the main MeshGraphNet model"""

import numpy as np
import torch

from meshnets.datatypes.graph import Graph
from meshnets.model.encoder import GraphEncoder
from meshnets.model.decoder import GraphDecoder
from meshnets.model.processor import GraphProcessor


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNet model.
    
    It is composed of a GraphEncoder, a GraphProcessor and a GraphDecoder.
    It takes as input Graph objects, encodes their node and mesh features
    to a latent representation, apply message passing steps
    on the latent Graph objects, and decodes the output features
    at each node."""

    def __init__(self, node_feats_size, mesh_feats_size, output_size,
                 latent_size, num_mlp_layers, message_passing_steps):
        """Initialize the GraphEncoder, GraphProcessor and GraphDecoder
        of the MeshGraphNet."""
        super().__init__()

        self.encoder = GraphEncoder(node_feats_size, mesh_feats_size,
                                    latent_size, num_mlp_layers)

        self.processor = GraphProcessor(latent_size, message_passing_steps)

        self.decoder = GraphDecoder(output_size, latent_size, num_mlp_layers)

    def forward(self, input: np.ndarray[Graph]) -> torch.Tensor:
        """Sequentially apply the MeshGraphNet to each Graph object
        in the input list.
        
        Return a list of the predictions Tensor for each Graph object."""

        # TODO(victor): normalize data

        output = []
        for graph in input:
            latent_graph = self.encoder(graph)
            latent_graph = self.processor(latent_graph)
            pred = self.decoder(latent_graph)
            output.append(pred)

        return output
