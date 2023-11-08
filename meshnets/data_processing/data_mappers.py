"""Function to process data from hugging face datasets to graphs.

This module assumes the data is in the format of the hugging face
datasets with the following columns:

- nodes: list of nodes in the graph with shape [n, 3] where 3
  corresponds to the spacial coordinates.

- edges: list of edges in the graph with shape [m, 2] where 2
  corresponds to the source and target nodes.

- wind_vector: list with shapes (3,) that contains the wind speeds on
  the 3 axis.

- wind_pressure: list with shape (N,) that contains the wind pressure
  on the nodes.

"""
import numpy as np


def to_undirected(example):
    """Makes the graph undirected.

    This is equivalent to adding the inverse edges to the graph. That
    is, if (i, j) is in the `edges` we also add the edge (j, i).

    """
    edges = np.array(example['edges'])
    if edges.size == 0:
        return example
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    example['edges'] = edges
    return example


def make_edge_features(example):
    """Makes the edge features from the node coordinates.

    The edge features:

    - The displacement vector between the source and target nodes.
    - The distance between the source and target nodes.

    """
    nodes = np.array(example['nodes'])
    edges = np.array(example['edges'])

    if edges.size == 0 or nodes.size == 0:
        example['edge_features'] = []
        return example
    displacement_vectors = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    distances = np.linalg.norm(displacement_vectors, axis=1)
    example['edge_features'] = np.concatenate(
        [displacement_vectors, distances[:, None]], axis=1)
    return example


def make_node_features(example, feature='wind_vector'):
    """Makes the node features from the node coordinates.

    The node features:

    - The wind vector. This is a common value for all the nodes in the
      graph.

    """
    example['node_features'] = [
        example[feature] for _ in range(len(example['nodes']))
    ]
    return example
