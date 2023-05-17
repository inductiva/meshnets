"""Process and format the mesh data for our model"""

from typing import Tuple

import numpy as np

from meshnets.datatypes.graph import EdgeSet
from meshnets.datatypes.graph import Graph


def _triangles_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the edges in a mesh of triangular faces.

    Each triangle edge is returned exactly twice, one for each direction.
    """

    edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, [0, 2]]],
                           axis=0)

    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]

    return (np.concatenate([senders,
                            receivers]), np.concatenate([receivers, senders]))


def _compute_mesh_features(nodes: np.ndarray, senders: np.ndarray,
                           receivers: np.ndarray) -> np.ndarray:
    """Compute mesh features for each edge.
    
    displacement vector : sender coords - receiver coords
    displacement norm : norm of the displacement vector
    """

    send_coords = nodes[senders]
    receive_coords = nodes[receivers]

    displacement = send_coords - receive_coords
    norm = np.linalg.norm(displacement, axis=1)
    mesh_features = np.column_stack([displacement, norm])

    return mesh_features


def triangle_mesh_to_graph(nodes: np.ndarray, node_features: np.ndarray,
                           faces: np.ndarray) -> Graph:
    """Produce a Graph object from triangle mesh data.

    Triangle edges exist twice in the Graph, once in each direction.
    """

    senders, receivers = _triangles_to_edges(faces)
    mesh_features = _compute_mesh_features(nodes, senders, receivers)
    edge_set = EdgeSet(mesh_features, senders, receivers)

    graph = Graph(node_features, edge_set)

    return graph
