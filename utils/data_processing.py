"""Process and format the mesh data for our model"""

from datatypes.graph import EdgeSet
from datatypes.graph import Graph

import numpy as np


def _triangles_to_edges(faces):

    edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, [0, 2]]],
                           axis=0)

    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]

    return (np.concatenate([senders,
                            receivers]), np.concatenate([receivers, senders]))


def _compute_mesh_features(nodes, senders, receivers):

    send_coords = nodes[senders]
    receive_coords = nodes[receivers]

    displacement = send_coords - receive_coords
    norm = np.linalg.norm(displacement, axis=1)
    mesh_features = np.column_stack([displacement, norm])

    return mesh_features


def mesh_to_graph(nodes: np.ndarray, node_features: np.ndarray,
                  cells: np.ndarray) -> Graph:

    senders, receivers = _triangles_to_edges(cells)
    mesh_features = _compute_mesh_features(nodes, senders, receivers)
    edge_set = EdgeSet(mesh_features, senders, receivers)

    graph = Graph(node_features, edge_set)

    return graph
