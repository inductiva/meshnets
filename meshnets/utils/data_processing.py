"""Process and format the mesh data for our model"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms.face_to_edge import FaceToEdge


def _compute_edge_attributes(data: Data, remove_pos: bool = True) -> Data:
    """Compute edge attributes for each edge of the graph.
    
    The edge attributes are the displacement vector between the two nodes
    of an edge and the norm of the displacement vector."""

    src, dst = data.edge_index

    src_pos = data.pos[src]
    dst_pos = data.pos[dst]

    displacement_vector = dst_pos - src_pos
    displacement_norm = torch.norm(displacement_vector, dim=1).unsqueeze(1)

    data.edge_attr = torch.cat([displacement_vector, displacement_norm], dim=1)

    if remove_pos:
        data.pos = None

    return data


def triangle_mesh_to_graph(node_coordinates: np.ndarray,
                           node_features: np.ndarray, faces: np.ndarray,
                           node_labels: np.ndarray) -> Data:
    """Produce an undirected graph object from triangle mesh data.
    
    The graph includes node features, edges indices, edge attributes
    and ground-truth label."""

    node_coordinates = torch.Tensor(node_coordinates)
    node_features = torch.Tensor(node_features)
    faces = torch.LongTensor(faces.T)
    node_labels = torch.Tensor(node_labels)

    mesh = Data(x=node_features,
                face=faces,
                y=node_labels,
                pos=node_coordinates)
    graph = FaceToEdge(remove_faces=True)(mesh)
    graph = _compute_edge_attributes(graph, remove_pos=False)

    return graph
