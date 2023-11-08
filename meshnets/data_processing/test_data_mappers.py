"""Test Hugging face processing functions."""
import numpy as np
import pytest

from . import data_mappers

example_3d = {
    'edges': [[0, 1], [1, 2], [1, 0], [2, 1]],
    'nodes': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
    'wind_vector': [1, 1, 1]
}
expected_edge_features_3d = np.array([[-1, 1, 0, np.sqrt(2)],
                                      [0, -1, 1, np.sqrt(2)],
                                      [1, -1, 0, np.sqrt(2)],
                                      [0, 1, -1, np.sqrt(2)]])
expected_node_features_3d = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                      [1, 1, 1]])

example_2d = {
    'edges': [[0, 1], [1, 2], [1, 0], [2, 1]],
    'nodes': [[1, 0], [0, 1], [0, 0], [1, 1]],
    'wind_vector': [1, 1]
}
expected_edge_features_2d = np.array([[-1, 1, np.sqrt(2)], [0, -1,
                                                            np.sqrt(1)],
                                      [1, -1, np.sqrt(2)], [0, 1,
                                                            np.sqrt(1)]])
expected_node_features_2d = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])

example_empty = {'edges': [], 'nodes': [], 'wind_vector': [1, 0, 0]}
expected_edge_features_empty = np.array([])
expected_node_features_empty = np.array([])


@pytest.mark.parametrize('edges, expected_edges',
                         [([[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2], [2, 3],
                                                      [1, 0], [2, 1], [3, 2]]),
                          ([[3, 4], [4, 5]], [[3, 4], [4, 5], [4, 3], [5, 4]]),
                          ([], [])])
def test_to_undirected(edges, expected_edges):
    example = {'edges': edges}
    # Call the function
    result = data_mappers.to_undirected(example)

    # Check if the edges have been properly modified
    assert np.array_equal(result['edges'], expected_edges)

    # Check if the original example object has not been modified
    assert np.array_equal(example['edges'],
                          expected_edges[:len(example['edges'])])


@pytest.mark.parametrize('example, expected_edge_features',
                         [(example_3d, expected_edge_features_3d),
                          (example_2d, expected_edge_features_2d),
                          (example_empty, expected_edge_features_empty)])
def test_make_edge_features(example, expected_edge_features):
    # Call the function
    result = data_mappers.make_edge_features(example)

    # Check if the features have been properly computed
    assert np.allclose(result['edge_features'],
                       expected_edge_features,
                       atol=1e-6)


@pytest.mark.parametrize('example, expected_node_features',
                         [(example_3d, expected_node_features_3d),
                          (example_2d, expected_node_features_2d),
                          (example_empty, expected_node_features_empty)])
def test_make_node_features(example, expected_node_features):
    result = data_mappers.make_node_features(example)
    assert np.allclose(result['node_features'],
                       expected_node_features,
                       atol=1e-6)
