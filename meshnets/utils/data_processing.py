"""Process and format the mesh data for our model"""

from typing import Optional
from typing import Tuple

from absl import logging
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms.face_to_edge import FaceToEdge

from meshnets.utils import data_loading


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


def edge_mesh_to_graph(node_coordinates: np.ndarray,
                       node_features: np.ndarray,
                       edges: np.ndarray,
                       node_labels: Optional[np.ndarray] = None) -> Data:
    """Produce an undirected graph object from edge mesh data and compute
    its edge features.

    Args shape:
        node_coordinates: (nb_nodes, nb_space_dim)
        node_features: (nb_nodes, nb_node_features)
        edges: (nb_edges, 2)
        node_labels: (nb_nodes, nb_node_labes)

    The graph includes node features, edges indices, edge attributes
    and optionally node labels."""

    node_coordinates = torch.tensor(node_coordinates, dtype=torch.float32)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Ensure a bidirectional graph with no duplicated edges
    #pylint: disable = no-value-for-parameter
    edge_index = to_undirected(torch.tensor(edges.T, dtype=torch.int64))

    if node_labels is not None:
        node_labels = torch.tensor(node_labels, dtype=torch.float32)

    graph = Data(x=node_features,
                 edge_index=edge_index,
                 y=node_labels,
                 pos=node_coordinates)

    graph = _compute_edge_attributes(graph, remove_pos=True)

    return graph


def triangle_mesh_to_graph(node_coordinates: np.ndarray,
                           node_features: np.ndarray,
                           faces: np.ndarray,
                           node_labels: Optional[np.ndarray] = None) -> Data:
    """Produce an undirected graph object from triangle mesh data and compute
    its edge features.

    Args shape:
        node_coordinates: (nb_nodes, nb_space_dim)
        node_features: (nb_nodes, nb_node_features)
        faces: (nb_faces, 3)
        node_labels: (nb_nodes, nb_node_labes)
    
    The graph includes node features, edges indices, edge attributes
    and optionally node labels."""

    node_coordinates = torch.tensor(node_coordinates, dtype=torch.float32)
    node_features = torch.tensor(node_features, dtype=torch.float32)
    faces = torch.tensor(faces.T, dtype=torch.int64)

    if node_labels is not None:
        node_labels = torch.tensor(node_labels, dtype=torch.float32)

    mesh = Data(x=node_features,
                face=faces,
                y=node_labels,
                pos=node_coordinates)
    graph = FaceToEdge(remove_faces=True)(mesh)
    graph = _compute_edge_attributes(graph, remove_pos=True)

    return graph


def mesh_file_to_graph_data(file_path: str,
                            wind_vector: Tuple[float],
                            get_pressure: bool = True,
                            verbose: bool = False) -> Data:
    """Receive the path to a mesh file (e.g. `.obj`, `.vtk` file) and
    a wind vector.
    
    Return a graph Data object with the following attributes:
        x: Node feature matrix
        edge_index: Graph connectivity in COO format
        edge_attr: Edge feature matrix
        y [Optional]: Graph-level or node-level ground-truth
    """

    if verbose:
        logging.info('Loading mesh data from %s', file_path)
    nodes, edges, pressure = data_loading.load_edge_mesh_pv(
        file_path, get_pressure=get_pressure, verbose=verbose)

    # node features for each node are the wind vector
    node_features = np.tile(wind_vector, (len(nodes), 1))

    if verbose:
        logging.info('Building graph from the mesh data')
    graph = edge_mesh_to_graph(nodes, node_features, edges, pressure)

    if verbose:
        logging.info('Node features shape : %s', graph.x.shape)
        logging.info('Edge index shape : %s', graph.edge_index.shape)
        logging.info('Edge features shape : %s', graph.edge_attr.shape)
        if graph.y is not None:
            logging.info('Pressure label shape : %s', graph.y.shape)

    return graph
