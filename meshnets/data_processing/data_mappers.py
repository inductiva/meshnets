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


def to_undirected(example, feature_key='edges'):
    """Makes the graph undirected.

    This is equivalent to adding the inverse edges to the graph. That
    is, if (i, j) is in the `edges` we also add the edge (j, i).

    """
    edges = np.array(example[feature_key])
    if edges.size == 0:
        return example
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    example[feature_key] = edges
    return example
