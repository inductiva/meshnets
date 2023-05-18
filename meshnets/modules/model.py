"""Define the main MeshGraphNet model"""

from typing import List

import torch

from meshnets.datatypes.graph import Graph
from meshnets.modules.encoder import GraphEncoder
from meshnets.modules.decoder import GraphDecoder
from meshnets.modules.processor import GraphProcessor


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNet model.
    
    The model architecture is composed of a GraphEncoder, a GraphProcessor
    and a GraphDecoder, and takes as input a list of Graph objects. Each Graph
    object's features are encoded to form a latent Graph representation. 
    Message passing steps are then applied by the Processor on the latent Graph
    object and the Decoder outputs features at each node from the processed
    latent node features."""

    def __init__(self, node_features_size, mesh_features_size, output_size,
                 latent_size, num_mlp_layers, message_passing_steps):
        """Initialize the GraphEncoder, GraphProcessor and GraphDecoder
        of the MeshGraphNet."""
        super().__init__()

        self.encoder = GraphEncoder(node_features_size, mesh_features_size,
                                    latent_size, num_mlp_layers)

        self.processor = GraphProcessor(latent_size, message_passing_steps)

        self.decoder = GraphDecoder(output_size, latent_size, num_mlp_layers)

    def forward(self, graph_list: List[Graph]) -> List[torch.Tensor]:
        """Sequentially apply the MeshGraphNet to each Graph object
        in the input list.
        
        Return a list of the predictions Tensor for each Graph object."""

        # TODO(victor): normalize data

        output_list = []
        for graph in graph_list:
            latent_graph = self.encoder(graph)
            latent_graph = self.processor(latent_graph)
            prediction = self.decoder(latent_graph)
            output_list.append(prediction)

        return output_list
