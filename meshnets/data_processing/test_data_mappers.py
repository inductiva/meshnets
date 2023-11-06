"""Test Hugging face processing functions."""
import numpy as np
import pytest

import data_mappers


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
